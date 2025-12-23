import io
import re
from PIL import Image, ImageDraw
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()


def refcoco_bbox_doc_to_visual(original_image, bbox):
    image = original_image.convert('RGB')
    draw = ImageDraw.Draw(image)
    bbox_xy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    draw.rectangle(bbox_xy, outline='red')
    return image.convert('RGB')


def refcoco_seg_doc_to_visual(original_image, segmentation):
    image = original_image.convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.polygon(segmentation)
    return image.convert('RGB')


@register_benchmark(
    BenchmarkMeta(
        name='refcoco',
        pretty_name='RefCOCO',
        description=(
            'The RefCOCO dataset is a dataset specifically '
            'designed for citation expression generation (REG) '
            'tasks, aimed at helping researchers better understand '
            'how to point to specific objects in images in natural '
            'language expressions.'
        ),
        tags=[Tags.KNOWLEDGE, Tags.MULTI_MODAL],
        dataset_id='lmms-lab/RefCOCO',
        metric_list=[
            'IoU', 'ACC@0.1', 'ACC@0.3', 'ACC@0.5', 'ACC@0.7', 'ACC@0.9', 'Center_ACC', 'Bleu_1', 'Bleu_2', 'Bleu_3',
            'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr'
        ],
        subset_list=['testA', 'testB'],
        extra_params={
            'eval_mode': {
                'type': 'str',
                'description': 'Control the evaluation mode used by RefCOCO.(bbox,seg,bbox_rec)',
                # 'value': 'bbox_rec'
            }
        }
    )
)
class RefCOCOAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_as_subset = True

        self.eval_mode = self.extra_params.get('eval_mode', 'bbox_rec')

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        image = record.get('image')
        original_bbox = record.get('bbox')
        content_list: List[Content] = []
        if self.eval_mode == 'bbox_rec':
            bbox_rec_template = """
            Bounding box coordinates are specified in the format
            (top-left x, top-left y, bottom-right x, bottom-right y).
            The values must be normalized according to the length and width of the image as required above.
            All values are floating point numbers bounded between 0 and 1.
            The content you return must only be in the format of [a, b, c, d], without any other content.
            This rule must be strictly followed.
            Please provide the bounding box coordinate of the region this sentence describes:
            """
            answer = record.get('answer')
            input_text = bbox_rec_template + answer[0]
        else:
            question = record.get('question', '')
            input_text = f"{question}\nAnswer the question using a single word or phrase."

        content_list.append(ContentText(text=input_text))
        bbox = []

        if image:
            image_data = Image.open(io.BytesIO(image['bytes']))
            if self.eval_mode == 'bbox_rec':
                image_width = image_data.width
                image_height = image_data.height
                bbox = [
                    original_bbox[0] / image_width, original_bbox[1] / image_height,
                    (original_bbox[0] + original_bbox[2]) / image_width,
                    (original_bbox[1] + original_bbox[3]) / image_height
                ]
                image_data = image_data.convert('RGB')
            elif self.eval_mode == 'bbox':
                image_data = refcoco_bbox_doc_to_visual(image_data, original_bbox)
            elif self.eval_mode == 'seg':
                segmentation = record.get('segmentation')
                image_data = refcoco_seg_doc_to_visual(image_data, segmentation)
            else:
                raise 'Invalid eval mode parameter'

            buffer = io.BytesIO()
            image_data.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()

            image_base64 = bytes_to_base64(image_bytes, format='JPEG', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        if self.eval_mode == 'bbox_rec':
            target = str(bbox)
        elif self.eval_mode in ['bbox', 'seg']:
            target = record.get('answer')
        else:
            raise 'Invalid eval mode parameter'

        metadata: Dict[str, Any] = {
            'question_id': record.get('question_id'),
            'iscrowd': record.get('iscrowd'),
            'file_name': record.get('file_name'),
            'answer': record.get('answer'),
            'original_bbox': original_bbox,
            'bbox': bbox,
            'eval_mode': self.eval_mode
        }

        return Sample(input=[ChatMessageUser(content=content_list)], target=target, metadata=metadata)

    def extract_answer(self, prediction: str, task_state: TaskState):
        if task_state.metadata['eval_mode'] == 'bbox_rec':
            # 匹配[a, b, c, d]格式的答案
            pattern = r'\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]'
            match = re.search(pattern, prediction)

            if match:
                return [float(match.group(i)) for i in range(1, 5)]

            return [0, 0, 0, 0]
        else:
            return prediction

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        # Initialize the score object with prediction details
        from evalscope.benchmarks.refcoco.evaluation_lib import process_results
        orig = str(original_prediction) if isinstance(original_prediction, list) else original_prediction
        filt = str(filtered_prediction) if isinstance(filtered_prediction, list) else filtered_prediction
        score = Score(
            extracted_prediction=filt,
            prediction=orig,
        )

        doc = task_state.metadata

        # try:
        results = process_results(doc, filtered_prediction)
        score.value.update(results)

        score.main_score_name = doc['eval_mode']

        # except Exception as e:
        #     logger.error(f'Error calculating ref_coco metrics: {e}')
        #     score.value = {}

        return score
