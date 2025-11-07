import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator.state import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'Regular Text Recognition', 'Irregular Text Recognition', 'Artistic Text Recognition', 'Handwriting Recognition',
    'Digit String Recognition', 'Non-Semantic Text Recognition', 'Scene Text-centric VQA', 'Doc-oriented VQA',
    'Key Information Extraction', 'Handwritten Mathematical Expression Recognition'
]


@register_benchmark(
    BenchmarkMeta(
        name='ocr_bench',
        pretty_name='OCRBench',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=
        'OCRBench is a comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models. It comprises five components: Text Recognition, SceneText-Centric VQA, Document-Oriented VQA, Key Information Extraction, and Handwritten Mathematical Expression Recognition. The benchmark includes 1000 question-answer pairs, and all the answers undergo manual verification and correction to ensure a more precise evaluation.',  # noqa: E501
        dataset_id='evalscope/OCRBench',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
    )
)
class OCRBenchAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_aggregation_name = False
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        input_text = self.prompt_template.format(question=record['question'])
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=json.dumps(record.get('answer'), ensure_ascii=False),  # answers is a list
            subset_key=record.get('question_type'),
            metadata={
                'dataset': record.get('dataset'),
                'question_type': record.get('question_type'),
            }
        )

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        pred = filtered_prediction.lower().strip()
        gt_ans = json.loads(reference)
        dataset_name = task_state.metadata['dataset']

        score_value = 0
        if dataset_name == 'HME100k':
            if isinstance(gt_ans, list):
                for j in range(len(gt_ans)):
                    answer = gt_ans[j].strip().replace('\n', ' ').replace(' ', '')
                    predict = pred.strip().replace('\n', ' ').replace(' ', '')
                    if answer in predict:
                        score_value = 1
            else:
                answer = gt_ans.strip().replace('\n', ' ').replace(' ', '')
                predict = pred.strip().replace('\n', ' ').replace(' ', '')
                if answer in predict:
                    score_value = 1
        else:
            if isinstance(gt_ans, list):
                for j in range(len(gt_ans)):
                    answer = gt_ans[j].lower().strip().replace('\n', ' ')
                    predict = pred.lower().strip().replace('\n', ' ')
                    if answer in predict:
                        score_value = 1
            else:
                answer = gt_ans.lower().strip().replace('\n', ' ')
                predict = pred.lower().strip().replace('\n', ' ')
                if answer in predict:
                    score_value = 1
        score.value = {'acc': score_value}
        return score
