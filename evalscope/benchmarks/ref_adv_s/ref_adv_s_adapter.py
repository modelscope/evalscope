# flake8: noqa: E501
import json
from typing import Any, Dict, List, Optional, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiTurnAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessage, ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.metrics.utils.functions import mean
from evalscope.utils.logger import get_logger

logger = get_logger()

DIRECT_PROMPT = (
    '<image>\n'
    'Locate every object that matches the description "{ref_sentence}" in the image. '
    'Report bbox coordinates in JSON format.'
)
COT_PROMPT = (
    '<image>\n'
    'Locate every object that matches the description "{ref_sentence}" in the image.\n'
    'Think first, then answer.\n'
    'Finally report bbox coordinates in JSON format.'
)
FOLLOWUP_PROMPT = (
    'Your previous answer did not end with a valid JSON bbox output.\\n'
    'Now output ONLY one complete JSON code block.\\n'
    'Use this exact schema:\\n'
    '```json\\n'
    '{"bboxes": [[x1, y1, x2, y2]]}\\n'
    '```\\n'
    'Do not output any extra text.'
)
PROMPTS = {'direct': DIRECT_PROMPT, 'cot': COT_PROMPT}

DESCRIPTION = """
## Overview

Ref-Adv-s is the public 1,142-case subset of Ref-Adv, a referring expression comprehension benchmark designed to test whether multimodal large language models can distinguish a target from hard same-category visual distractors instead of relying on grounding shortcuts.

## Task Description

- **Task Type**: Referring expression comprehension / visual grounding
- **Input**: One image and an English referring expression
- **Output**: One or more bounding boxes in JSON, with the first box used for scoring
- **Domain**: COCO and OpenImages scenes containing hard same-category distractors

## Key Features

- Contains 1,142 public cases sampled from the 5,000-case Ref-Adv benchmark
- Includes human-authored and model-assisted expressions, explicit negation, and at least two distractors per case
- Preserves the official `direct` and chain-of-thought (`cot`) prompt modes
- Uses the dataset's single `train` split as the evaluation split

## Evaluation Notes

- Reports official `Acc@0.5`, `Acc@0.75`, and `Acc@0.9` metrics from the IoU of the first parsed box
- Also reports `Acc@0.5` for the official distractor-count bins `2-3`, `4-6`, and `>=7`
- Parses the last valid fenced JSON object, or an unfenced JSON value that ends the response, using the official key search order
- A failed first parse triggers the official one-turn format-repair prompt; a second failure receives zero accuracy
- Set `pred_box_format` to `abs_xyxy` for Qwen2.5-VL and to `norm_1000_xyxy` for Qwen3-VL/Qwen3.5; `norm_1_xyxy` is also supported by the official evaluator
- [Paper](https://arxiv.org/abs/2602.23898) | [GitHub](https://github.com/dddraxxx/Ref-Adv)
"""


@register_benchmark(
    BenchmarkMeta(
        name='ref_adv_s',
        pretty_name='Ref-Adv-s',
        dataset_id='evalscope/ref-adv-s',
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2602.23898',
        tags=[Tags.MULTI_MODAL, Tags.GROUNDING, Tags.REASONING],
        subset_list=['default'],
        eval_split='train',
        metric_list=[
            'ACC@0.5',
            'ACC@0.75',
            'ACC@0.9',
            '2-3/ACC@0.5',
            '4-6/ACC@0.5',
            '>=7/ACC@0.5',
        ],
        primary_metric=MetricSelector(
            name='accuracy', aggregation='mean', dimensions={'scope': 'overall', 'threshold': 0.5}
        ),
        prompt_template=DIRECT_PROMPT,
        extra_params={
            'prompt_mode': {
                'type': 'str',
                'description': 'Official prompt mode.',
                'value': 'direct',
                'choices': ['direct', 'cot'],
            },
            'pred_box_format': {
                'type': 'str',
                'description': 'Coordinate format emitted by the evaluated model.',
                'value': 'norm_1000_xyxy',
                'choices': ['abs_xyxy', 'norm_1000_xyxy', 'norm_1_xyxy'],
            },
        },
        evaluation_version='v1.0',
    )
)
class RefAdvSAdapter(VisionLanguageAdapter, MultiTurnAdapter):
    """Adapter for the public Ref-Adv-s referring expression benchmark."""

    max_turns = 2

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prompt_template = PROMPTS[self.extra_params['prompt_mode']]

    def record_to_sample(self, record: Dict[str, Any]) -> Optional[Sample]:
        """Convert one Ref-Adv-s record into a multimodal grounding sample."""
        from evalscope.utils.io_utils import PIL_to_base64

        from .utils import base64_image_size, to_normalized_xyxy

        image_field = record.get('image')
        if isinstance(image_field, dict) and image_field.get('bytes'):
            image_uri = self._image_bytes_to_base64(image_field['bytes'], default_format='jpeg')
        elif hasattr(image_field, 'save'):
            image_uri = PIL_to_base64(image_field.convert('RGB'), format='JPEG', add_header=True)
        else:
            logger.warning(f'Ref-Adv-s row {record.get("row_idx")} has no usable image; skipping.')
            return None

        width, height = int(record['width']), int(record['height'])
        sent_size = [width, height]
        if self._max_image_bytes is not None:
            sent_size = list(base64_image_size(image_uri))

        target_box = record.get('solution')
        if not isinstance(target_box, list) or len(target_box) != 4:
            raise ValueError(f'Ref-Adv-s row {record.get("row_idx")} has an invalid solution: {target_box!r}')
        target_normalized = to_normalized_xyxy(target_box, (width, height), 'abs_xyxy')
        distractor_count = int(record['distractors'])
        prompt_text = self.prompt_template.format(ref_sentence=record['normal_caption'])
        content: List[Content] = [ContentText(text=prompt_text), ContentImage(image=image_uri)]

        return Sample(
            input=[ChatMessageUser(content=content)],
            target=json.dumps(target_box),
            metadata={
                'row_idx': record['row_idx'],
                'file_name': record['file_name'],
                'image_source': record['image_source'],
                'human_authored': record['human_authored'],
                'use_negation': record['use_negation'],
                'distractor_count': distractor_count,
                'target_box_normalized': target_normalized,
                'sent_size': sent_size,
                'retry_followup_used': False,
            },
        )

    def initialize_history(self, sample: Sample) -> List[ChatMessage]:
        """Keep any configured system message outside the turn-building loop."""
        return list(sample.input[:-1])

    def build_turn_prompt(
        self,
        sample: Sample,
        history: List[ChatMessage],
        turn_index: int,
    ) -> Optional[Union[str, ChatMessage]]:
        """Issue the official prompt, then retry once only when JSON parsing fails."""
        if turn_index == 0:
            return sample.input[-1]

        from .utils import parse_bboxes

        boxes, _ = parse_bboxes(
            history[-1].text,
            image_size=sample.metadata['sent_size'],
            box_format=self.extra_params['pred_box_format'],
        )
        if boxes:
            return None
        sample.metadata['retry_followup_used'] = True
        return FOLLOWUP_PROMPT

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the first official JSON bbox and normalize it to unit coordinates."""
        from .utils import parse_bboxes

        boxes, parse_error = parse_bboxes(
            prediction,
            image_size=task_state.metadata['sent_size'],
            box_format=self.extra_params['pred_box_format'],
        )
        task_state.metadata['parse_error'] = parse_error
        return json.dumps(boxes[0]) if boxes else ''

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Compute the official first-box IoU threshold metrics and distractor breakdown."""
        from .utils import distractor_bin, iou_xyxy

        prediction_box = json.loads(filtered_prediction) if filtered_prediction else None
        target_box = task_state.metadata['target_box_normalized']
        iou = iou_xyxy(prediction_box, target_box) if prediction_box is not None else 0.0
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = {
            'ACC@0.5': float(iou >= 0.5),
            'ACC@0.75': float(iou >= 0.75),
            'ACC@0.9': float(iou >= 0.9),
        }
        bin_name = distractor_bin(task_state.metadata['distractor_count'])
        if bin_name is not None:
            score.value[f'{bin_name}/ACC@0.5'] = float(iou >= 0.5)
        score.main_score_name = 'ACC@0.5'
        score.metadata = {
            'iou': iou,
            'parse_error': task_state.metadata.get('parse_error', ''),
            'retry_followup_used': task_state.metadata.get('retry_followup_used', False),
        }
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate official thresholds and distractor bins as structured accuracy metrics."""
        metric_specs = (
            ('ACC@0.5', 0.5, 'overall'),
            ('ACC@0.75', 0.75, 'overall'),
            ('ACC@0.9', 0.9, 'overall'),
            ('2-3/ACC@0.5', 0.5, '2-3'),
            ('4-6/ACC@0.5', 0.5, '4-6'),
            ('>=7/ACC@0.5', 0.5, '>=7'),
        )
        aggregates = []
        for raw_name, threshold, scope in metric_specs:
            selected = [sample_score for sample_score in sample_scores if raw_name in sample_score.score.value]
            if not selected:
                continue
            aggregates.append(
                AggScore(
                    score=mean([float(item.score.value[raw_name]) for item in selected]),
                    metric_name='accuracy',
                    aggregation='mean',
                    dimensions={'scope': scope, 'threshold': threshold},
                    num=len(selected),
                    ids=[item.sample_id for item in selected],
                )
            )
        return aggregates
