# flake8: noqa: E501
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, prompt

logger = get_logger()

# 定义提示模板
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".lstrip()

SUBSET_LIST = ['default']


@register_benchmark(
    BenchmarkMeta(
        name='zerobench',
        pretty_name='ZeroBench',
        dataset_id='evalscope/zerobench',
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.MULTI_MODAL],
        description='ZeroBench is a challenging visual reasoning benchmark for Large Multimodal Models (LMMs).',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='zerobench',
        train_split='zerobench_subquestions',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class ZeroBenchAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question_text']
        content_list: List[Content] = [ContentText(text=PROMPT_TEMPLATE.format(question=question))]
        image = record['question_images_decoded']
        if len(image) > 0:
            for img in image:
                image_base64 = bytes_to_base64(img['bytes'], format='jpeg', add_header=True)
                content_list.append(ContentImage(image=image_base64))

        metadata = {
            'question_id': record['question_id'],
            'question_images': record['question_images'],
            'image_attribution': record['image_attribution']
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)], target=record['question_answer'], metadata=metadata
        )
