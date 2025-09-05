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

SUBSET_LIST = ['default']

OPEN_PROMPT = """
Solve the following problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
"""

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

MULTI_CHOICE_TYPE = 'multi_choice'
OPEN_TYPE = 'free_form'


@register_benchmark(
    BenchmarkMeta(
        name='math_vista',
        pretty_name='MathVista',
        dataset_id='evalscope/MathVista',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'MathVista is a consolidated Mathematical reasoning benchmark within Visual contexts. It consists of three newly created datasets, IQTest, FunctionQA, and PaperQA, which address the missing visual domains and are tailored to evaluate logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively. It also incorporates 9 MathQA datasets and 19 VQA datasets from the literature, which significantly enrich the diversity and complexity of visual perception and mathematical reasoning challenges within our benchmark. In total, MathVista includes 6,141 examples collected from 31 different datasets.',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='testmini',
        prompt_template=OPEN_PROMPT,
    )
)
class MathVistaAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        content_list, answers_list = MathVistaAdapter.create_content_and_answers_list(record)

        if record['question_type'] == 'multi_choice':
            label_answer = self.get_option_label(answers_list, record['answer'])
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                choices=answers_list,
                target=label_answer,
                metadata={
                    'question_type': record['question_type'],
                    'answer_type': record['answer_type'],
                    **record['metadata'],
                }
            )
        elif record['question_type'] == 'free_form':
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=record['answer'],
                metadata={
                    'precision': record['precision'],
                    'question_type': record['question_type'],
                    'answer_type': record['answer_type'],
                    **record['metadata'],
                }
            )
        else:
            raise ValueError(f"Unexpected question_type: {record['question_type']}")

    def get_option_label(self, options, value):
        try:
            index = options.index(value)
            return chr(ord('A') + index)
        except ValueError:
            logger.warning(f"Answer '{value}' not found in options: {options}. This may cause evaluation issues.")
            return value

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
            answers_list = record['choices']
            input_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)
            content_list: list[Content] = [ContentText(text=input_text)]
        else:
            answers_list: list[str] = []
            content_list: list[Content] = [ContentText(text=OPEN_PROMPT.format(question=record['question']))]
        image = record['decoded_image']
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        return content_list, answers_list
