# flake8: noqa: E501
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import parse_answers

logger = get_logger()

MULT_CHOICE_PROMPT = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A, B, C, D. Think step by step before answering.

{question}
"""

SUBSET_LIST = [
    'Quantitative Reasoning', 'Other', 'Positional Reasoning', 'Stylistic Reasoning', 'Spatial Reasoning',
    'Attribute Reasoning'
]


@register_benchmark(
    BenchmarkMeta(
        name='visu_logic',
        pretty_name='VisuLogic',
        dataset_id='evalscope/VisuLogic',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description='A Challenging Visual-centric Benchmark for Evaluating Multimodal Reasoning in MLLMs!',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class VisuLogicAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        content_list: list[Content] = []
        prompt_text = MULT_CHOICE_PROMPT.format(question=question).strip()
        content_list.append(ContentText(text=prompt_text))

        image = record.get('image')
        if image and isinstance(image, dict):
            image_bytes = image.get('bytes')
            if image_bytes:
                image_base64 = bytes_to_base64(image_bytes, format='png', add_header=True)
                content_list.append(ContentImage(image=image_base64))

        metadata = {
            'id': record['id'],
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['label'],
            subset_key=record['tag'],
            metadata=metadata,
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        answers = parse_answers(task_state)
        return ''.join(sorted(list(answers)))
