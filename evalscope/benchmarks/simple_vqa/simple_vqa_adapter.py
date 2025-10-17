# flake8: noqa: E501
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger


logger = get_logger()

SUBSET_LIST = ['default']

# Define prompt template
PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".lstrip()

@register_benchmark(
    BenchmarkMeta(
        name='simple_vqa',
        pretty_name='SimpleVQA',
        dataset_id='m-a-p/SimpleVQA',
        tags=[Tags.REASONING],
        description=
        'SimpleVQA, the first comprehensive multi-modal benchmark to evaluate the factuality ability of MLLMs to answer natural language short questions.',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class SimpleVQAAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list: list[Content] = [ContentText(text=PROMPT_TEMPLATE.format(question=record['question']))]
        image = record['image']
        if image:
            content_list.append(ContentImage(image=image))
        return Sample(
             input=[ChatMessageUser(content=content_list)],
             target=record['answer'],
             metadata={
                'data_id': record['data_id'],
                'image_description': record['image_description'],
                'language':record['language'],
                'original_category': record['original_category'],
                'source': record['source'],
                'atomic_question': record['atomic_question'],
                'atomic_fact': record['atomic_fact'],
                # **record['vqa_category']
             }
        )
    
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        pattern = r'ANSWER:\s*(.*)'
        match = re.search(pattern, prediction)
        if match:
            return match.group(1).strip()
        return ''