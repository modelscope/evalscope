import io
import re
from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


class EvalMode:
    """Evaluation modes for RefCOCO benchmark."""
    BBOX = 'bbox'  # Image Captioning with BBox visualization
    SEG = 'seg'  # Image Captioning with Segmentation visualization
    BBOX_REC = 'bbox_rec'  # Referring Expression Comprehension (REC)


BBOX_REC_TEMPLATE = """
Bounding box coordinates are specified in the format (top-left x, top-left y, bottom-right x, bottom-right y).
The values must be normalized according to the length and width of the image as required above. All values are floating point numbers bounded between 0 and 1. The content you return must only be in the format of [a, b, c, d], without any other content.
This rule must be strictly followed. Please provide the bounding box coordinate of the region this sentence describes:
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='refcoco',
        pretty_name='RefCOCO',
        description=
        'The RefCOCO dataset is a collection of images, object bounding boxes, and free-form natural-language referring expressions intended for training and evaluating models on the task of Referring Expression Comprehension (REC). It was created by collecting expressions on Amazon Mechanical Turk that uniquely describe a target object inside a MSCOCO image, and then asking other Turkers to click on the corresponding object.',  # noqa: E501
        tags=[Tags.KNOWLEDGE, Tags.MULTI_MODAL, Tags.GROUNDING, Tags.IMAGE_CAPTIONING],
        dataset_id='lmms-lab/RefCOCO',
        metric_list=[
            'IoU', 'ACC@0.1', 'ACC@0.3', 'ACC@0.5', 'ACC@0.7', 'ACC@0.9', 'Center_ACC', 'Bleu_1', 'Bleu_2', 'Bleu_3',
            'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'
        ],
        subset_list=['test', 'val', 'testA', 'testB'],
        extra_params={
            'eval_mode': {
                'type':
                'str',
                'description':
                'Control the evaluation mode used by RefCOCO. '
                'bbox: image caption task, visualize the original image with bounding box; '
                'seg: image caption task, visualize the original image with segmentation; '
                'bbox_rec: grounding task, recognize bounding box coordinates.',
                'value':
                EvalMode.BBOX,
                'choices': [EvalMode.BBOX, EvalMode.SEG, EvalMode.BBOX_REC]
            }
        }
    )
)
class RefCOCOAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.eval_mode = self.extra_params.get('eval_mode', EvalMode.BBOX)

        check_import(module_name='pycocoevalcap', package='pycocoevalcap', raise_error=True)

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        from PIL import Image

        from evalscope.benchmarks.refcoco.utils import refcoco_bbox_doc_to_visual, refcoco_seg_doc_to_visual

        image_info = record.get('image', {})
        image_bytes = image_info.get('bytes') if image_info else None
        original_bbox = record.get('bbox')
        segmentation = record.get('segmentation')

        bbox_norm = []
        image_base64 = None

        if image_bytes:
            image_data = Image.open(io.BytesIO(image_bytes))

            # Process image based on evaluation mode
            if self.eval_mode == EvalMode.BBOX_REC:
                # Normalize bbox: [x, y, w, h] -> [x1/W, y1/H, x2/W, y2/H]
                w, h = image_data.size
                x, y, bw, bh = original_bbox
                bbox_norm = [x / w, y / h, (x + bw) / w, (y + bh) / h]
                image_data = image_data.convert('RGB')
            elif self.eval_mode == EvalMode.BBOX:
                image_data = refcoco_bbox_doc_to_visual(image_data, original_bbox)
            elif self.eval_mode == EvalMode.SEG:
                image_data = refcoco_seg_doc_to_visual(image_data, segmentation)
            else:
                raise ValueError(f'Invalid eval mode: {self.eval_mode}')

            # Convert processed image to base64
            buffer = io.BytesIO()
            image_data.save(buffer, format='JPEG')
            image_base64 = bytes_to_base64(buffer.getvalue(), format='JPEG', add_header=True)

        metadata: Dict[str, Any] = {
            'question_id': record.get('question_id'),
            'iscrowd': record.get('iscrowd'),
            'file_name': record.get('file_name'),
            'answer': record.get('answer'),
            'original_bbox': original_bbox,
            'bbox': bbox_norm,
            'eval_mode': self.eval_mode
        }

        if self.eval_mode == EvalMode.BBOX_REC:
            return self._create_bbox_rec_samples(record, image_base64, bbox_norm, metadata)
        else:
            return self._create_caption_sample(record, image_base64, metadata)

    def _create_bbox_rec_samples(self, record: Dict, image_base64: str, bbox_norm: List[float],
                                 metadata: Dict) -> List[Sample]:
        """Create samples for Bounding Box Recognition (REC) task."""
        answers = record.get('answer', [])
        target = str(bbox_norm)
        samples = []

        for ans in answers:
            input_text = BBOX_REC_TEMPLATE + ans
            content_list = [ContentText(text=input_text)]
            if image_base64:
                content_list.append(ContentImage(image=image_base64))

            samples.append(Sample(input=[ChatMessageUser(content=content_list)], target=target, metadata=metadata))
        return samples

    def _create_caption_sample(self, record: Dict, image_base64: str, metadata: Dict) -> Sample:
        """Create sample for Image Captioning (REG) task."""
        question = record.get('question', '')
        input_text = f'{question}\nAnswer the question using a single word or phrase.'
        target = str(record.get('answer'))

        content_list = [ContentText(text=input_text)]
        if image_base64:
            content_list.append(ContentImage(image=image_base64))

        return Sample(input=[ChatMessageUser(content=content_list)], target=target, metadata=metadata)

    def extract_answer(self, prediction: str, task_state: TaskState) -> Union[str, List[float]]:
        """Extract answer from model prediction."""
        if task_state.metadata['eval_mode'] == EvalMode.BBOX_REC:
            # Match [a, b, c, d] pattern for bounding box coordinates
            pattern = r'\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]'
            match = re.search(pattern, prediction)

            if match:
                return [float(match.group(i)) for i in range(1, 5)]
            return [0.0, 0.0, 0.0, 0.0]
        else:
            return prediction

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        from evalscope.benchmarks.refcoco.evaluation_lib import process_results

        orig = str(original_prediction) if isinstance(original_prediction, list) else original_prediction
        filt = str(filtered_prediction) if isinstance(filtered_prediction, list) else filtered_prediction

        score = Score(
            extracted_prediction=filt,
            prediction=orig,
        )

        doc = task_state.metadata
        results = process_results(doc, filtered_prediction)
        score.value.update(results)

        # Set main score name based on eval mode (e.g. for grouping in reports)
        score.main_score_name = doc['eval_mode']

        return score
