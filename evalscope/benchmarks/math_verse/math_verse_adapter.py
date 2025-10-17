# flake8: noqa: E501
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, prompt

logger = get_logger()

MULTI_CHOICE_TYPE = 'multi-choice'
OPEN_TYPE = 'free-form'

OPEN_PROMPT = """
Solve the following problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
"""

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

SUBSET_LIST = ['testmini']

@register_benchmark(
    BenchmarkMeta(
        name='math_verse',
        pretty_name='MathVerse',
        dataset_id='evalscope/MathVerse',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'MathVerse, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. ',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='testmini',
        prompt_template=OPEN_PROMPT,
    )
)
class MathVerseAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list, answers_list = MathVerseAdapter.create_content_and_answers_list(record)
        question_type = record['question_type']

        if record['question_type'] == 'multi-choice':
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                choices=answers_list,
                target=record['answer'],
                metadata={
                    'sample_index': record['sample_index'],
                    'problem_index': record['problem_index'],
                    'problem_version': record['problem_version'],
                    'question_type': record['question_type'],
                    'query_wo': record['query_wo'],
                    'query_cot': record['query_cot'],
                    'question_for_eval': record['question_for_eval'],
                    **record['metadata'],
                }
            )
        elif record['question_type'] == 'free-form':
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=record['answer'],
                metadata={
                    'sample_index': record['sample_index'],
                    'problem_index': record['problem_index'],
                    'problem_version': record['problem_version'],
                    'question_type': record['question_type'],
                    'query_wo': record['query_wo'],
                    'query_cot': record['query_cot'],
                    'question_for_eval': record['question_for_eval'],
                    **record['metadata'],
                }
            )
        else:
            raise ValueError(f"Unexpected question_type: {record['question_type']}")

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        question_type = task_state.metadata['question_type']
        if question_type == MULTI_CHOICE_TYPE:
            answers = parse_answers(task_state)
            return ''.join(sorted(list(answers)))
        elif question_type == OPEN_TYPE:
            pattern = r'ANSWER:\s*(.*)'
            match = re.search(pattern, prediction)
            if match:
                return match.group(1).strip()
            return ''
        else:
            raise ValueError(f'Unsupported question type: {question_type}')

    @staticmethod
    def create_content_and_answers_list(record: dict[str, Any], ) -> tuple[list[Content], list[str]]:
        """
            Create a list of content elements and a list of answers from a record.

            Args:
                record (dict): The record containing question, images, and options.


            Returns:
                tuple: A tuple containing:
                    - content_list (list): A list of content elements (text and images).
                    - answers_list (list): A list of possible answers (for multiple-choice questions).
        """
        question_type = record['question_type']
        if question_type == MULTI_CHOICE_TYPE:
            answers_list = MathVerseAdapter.get_answers_list(record)
            input_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT) 
            content_list: list[Content] = [ContentText(text=input_text)]
        else:
            answers_list: list[str] = []
            content_list: list[Content] = [ContentText(text=OPEN_PROMPT.format(question=record['question']))]
        image = record['image']
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        return content_list, answers_list

    @staticmethod
    def get_answers_list(record: dict[str, Any], ) -> list[str]:
        text = record['question']
        parts = re.split(r'Choices:\s*', text, flags=re.IGNORECASE)
        if len(parts) < 2:
            return []
        content = parts[1]
        # Match all option lines
        pattern = re.compile(r'^([A-Z])\s*:\s*(.+?)\s*$', re.MULTILINE)
        matches = pattern.findall(content)

        return [match[1] for match in matches]