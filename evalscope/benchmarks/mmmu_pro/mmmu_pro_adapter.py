import ast
from typing import Any, Dict, List

from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, prompt
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.api.dataset import Sample
from evalscope.constants import Tags
from evalscope.api.messages import ContentImage, ContentText, Content, ChatMessageUser

logger = get_logger()

SUBSET_LIST = [
    'standard (4 options)',
    'standard (10 options)'
]

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pro',
        pretty_name='MMMU_PRO',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description='MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the true understanding capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU benchmark by introducing several key improvements that make it more challenging and realistic, ensuring that models are evaluated on their genuine ability to integrate and comprehend both visual and textual information.',
        dataset_id='AI-ModelScope/MMMU_Pro',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT
    )
)
class MMMUPROAdapter(DefaultDataAdapter):
    MAX_IMAGES: int = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        metadata = {
            'id': record['id'],
            'explanation': record['explanation'],
            'img_type': record['img_type'],
            'topic_difficulty':record['topic_difficulty'],
            'subject': record['subject']
        }
        answers_list: List[str] = ast.literal_eval(record['options'])
        input_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)
        content_list: List[Content] = [ContentText(text=input_text)]

        for i in range(MMMUPROAdapter.MAX_IMAGES):
            image = record[f'image_{i+1}']
            if image:
                image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
                content_list.append(ContentImage(image=image_base64))
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=record['answer'],
            metadata=metadata,
        )
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))