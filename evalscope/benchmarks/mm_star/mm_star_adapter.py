import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

MULT_CHOICE_PROMPT = r"""
Answer the following multiple choice question.
The last line of your response should be of the following format:
'ANSWER: [LETTER]' (without quotes)
where [LETTER] is one of A,B,C,D. Think step by step before answering.

{question}
""".strip()

SUBSET_LIST = [
    'coarse perception', 'fine-grained perception', 'instance reasoning', 'logical reasoning', 'math',
    'science & technology'
]


@register_benchmark(
    BenchmarkMeta(
        name='mm_star',
        pretty_name='MMStar',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=
        'MMStar: an elite vision-indispensible multi-modal benchmark, aiming to ensure each curated sample exhibits visual dependency, minimal data leakage, and requires advanced multi-modal capabilities.',  # noqa: E501
        dataset_id='evalscope/MMStar',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        default_subset='val',
        eval_split='val',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class MMStarAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        input_text = MULT_CHOICE_PROMPT.format(question=record['question'])
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        label_answer = record.get('answer')
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=['A', 'B', 'C', 'D'],
            target=label_answer,
            subset_key=record.get('category'),
            metadata={
                'index': record.get('index'),
                'category': record.get('category'),
                'l2_category': record.get('l2_category'),
                'source': record.get('meta_info', {}).get('source'),
                'split': record.get('meta_info', {}).get('split'),
                'image_path': record.get('meta_info', {}).get('image_path')
            }
        )
