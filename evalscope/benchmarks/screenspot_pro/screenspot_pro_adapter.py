# flake8: noqa: E501
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from .utils import AUTO, COORDINATE_SPACES

logger = get_logger()

DESCRIPTION = """
## Overview

ScreenSpot-Pro is a GUI grounding benchmark built from authentic high-resolution screenshots of \
professional desktop software. Given a natural-language instruction, a model must locate the target \
UI element on the screen, which stresses fine-grained localization on large, densely populated displays.

## Task Description

- **Task Type**: GUI grounding (single click-point prediction)
- **Input**: A full-resolution desktop screenshot + an English instruction describing the target UI element
- **Output**: One click point `[x, y]` in normalized coordinates within [0, 1]
- **Domain**: Professional desktop applications across CAD, Creative, Dev, Office, OS and Scientific software

## Key Features

- 1,581 expert-annotated instructions over 26 applications and 3 platforms (Windows, macOS, Linux)
- Screenshots are genuinely high-resolution (up to 6016x3384), so target elements often occupy well under 0.1% of the image
- Samples are grouped into six professional domains (`CAD`, `Creative`, `Dev`, `OS`, `Office`, `Scientific`), each exposed as a subset
- Every element is labelled as `text` or `icon`, enabling separate reporting for textual versus iconographic targets
- Ground-truth boxes are pixel coordinates paired with the original image size, and are normalized before scoring

## Evaluation Notes

- Primary metric: **acc** — a prediction is correct when the predicted point falls inside the ground-truth bounding box
- Secondary metrics: **text_acc** and **icon_acc**, each averaged over the samples of the corresponding `ui_type`
- Predictions are parsed from `[x, y]` points, `x=.., y=..` pairs or bounding boxes (reduced to their center), always taking the *last* match so that restated prompts and step-by-step reasoning do not shadow the final answer
- Ground truth is normalized to [0, 1], and predictions are mapped into the same space according to `coordinate_space`. With the default `auto`, values in [0, 1] are taken as normalized, values up to 1000 as the thousandths grid many VLMs emit, and larger values as pixels of the image the model received. Pin `coordinate_space` explicitly when a model's convention is known, since the 1–1000 range is inherently ambiguous
- The dataset ships a single `train` split, which is used as the evaluation split
- Images are large; `max_image_bytes` in `dataset_args` can cap the request size, and pixel-space predictions are normalized with the size of the image actually sent
- [Paper](https://arxiv.org/abs/2504.07981) | [GitHub](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)
"""

PROMPT_TEMPLATE = (
    'Identify the UI element for the instruction and output exactly one click point as [x, y] '
    'in normalized coordinates within [0, 1]. Do not output a bounding box.\n'
    'Instruction: {instruction}'
)


@register_benchmark(
    BenchmarkMeta(
        name='screenspot_pro',
        pretty_name='ScreenSpot-Pro',
        dataset_id='lmms-lab/ScreenSpot-Pro',
        tags=[Tags.MULTI_MODAL, Tags.GROUNDING, Tags.AGENT],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2504.07981',
        subset_list=['CAD', 'Creative', 'Dev', 'OS', 'Office', 'Scientific'],
        metric_list=['acc', 'text_acc', 'icon_acc'],
        eval_split='train',
        extra_params={
            'coordinate_space': {
                'type': 'str',
                'description': 'Coordinate convention used to interpret predicted click points. '
                'auto: infer from the magnitude of the coordinates; '
                'normalized: values already in [0, 1]; '
                'thousandths: values on a 0-1000 grid; '
                'pixel: pixels of the image sent to the model.',
                'value': AUTO,
                'choices': list(COORDINATE_SPACES),
            }
        },
    )
)
class ScreenSpotProAdapter(VisionLanguageAdapter):
    """Data adapter for lmms-lab/ScreenSpot-Pro.

    Samples are regrouped by their professional domain (``group``) into top-level subsets.
    Scoring is deterministic: the predicted click point must fall inside the ground-truth
    bounding box, with both mapped into normalized [0, 1] coordinates.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Group samples into subsets by their professional domain
        self.reformat_subset = True
        self.coordinate_space = self.extra_params.get('coordinate_space', AUTO)

    def record_to_sample(self, record: Dict[str, Any]) -> Optional[Sample]:
        """Convert a raw ScreenSpot-Pro record to a multimodal Sample."""
        from .utils import base64_image_size, normalize_bbox

        image_field = record.get('image')
        if not isinstance(image_field, dict) or not image_field.get('bytes'):
            logger.warning(f'Record {record.get("id")} has no usable image; skipping.')
            return None

        width, height = record['img_size']
        bbox_norm = normalize_bbox(record['bbox'], width, height)

        image_b64 = self._image_bytes_to_base64(image_field['bytes'], default_format='png')
        content_list: List[Content] = [
            ContentImage(image=image_b64),
            ContentText(text=PROMPT_TEMPLATE.format(instruction=record.get('instruction', ''))),
        ]

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=str([round(coord, 4) for coord in bbox_norm]),
            subset_key=record.get('group', ''),
            metadata={
                'id': record.get('id', ''),
                # Size of the image actually sent, which max_image_bytes may have shrunk
                'sent_size': list(base64_image_size(image_b64)),
                'bbox_norm': bbox_norm,
                'ui_type': record.get('ui_type', ''),
                'application': record.get('application', ''),
                'platform': record.get('platform', ''),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the predicted click point, mapped onto normalized [0, 1] coordinates."""
        from .utils import parse_point, to_normalized_point

        point = parse_point(prediction)
        if point is None:
            return ''

        point = to_normalized_point(point, task_state.metadata['sent_size'], self.coordinate_space)
        return f'[{point[0]:.4f}, {point[1]:.4f}]'

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score a prediction by checking whether the click point hits the target element.

        Returns a Score with ``acc`` plus ``text_acc`` or ``icon_acc`` depending on the
        sample's ``ui_type``, so that both breakdowns are averaged over their own samples.
        """
        from .utils import parse_point, point_in_bbox

        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)

        point = parse_point(filtered_prediction) if filtered_prediction else None
        metadata = task_state.metadata or {}
        correct = float(point is not None and point_in_bbox(point, metadata['bbox_norm']))

        score.value = {'acc': correct}
        ui_type = metadata.get('ui_type', '')
        if ui_type in ('text', 'icon'):
            score.value[f'{ui_type}_acc'] = correct
        score.main_score_name = 'acc'
        return score
