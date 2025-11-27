# flake8: noqa: E501
import re
from typing import Any, Dict, List

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

OPEN_PROMPT = '{question}\nPlease reason step by step, and put your final answer within \\boxed{{}} without units.'

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

SUBSET_LIST = ['level 1', 'level 2', 'level 3', 'level 4', 'level 5']


@register_benchmark(
    BenchmarkMeta(
        name='math_vision',
        pretty_name='MathVision',
        dataset_id='evalscope/MathVision',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'The MATH-Vision (MATH-V) dataset, a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions.',
        subset_list=SUBSET_LIST,
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        eval_split='test',
        prompt_template=OPEN_PROMPT,
    )
)
class MathVisionAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        if len(record['options']) > 0:
            question_type = 'multi_choice'
        else:
            question_type = 'free_form'
        content_list, answers_list = MathVisionAdapter.create_content_and_answers_list(record, question_type)
        metadata = {
            'id': record['id'],
            'image': record['image'],
            'solution': record['solution'],
            'level': record['level'],
            'question_type': question_type,
            'subject': record['subject']
        }
        if question_type == 'multi_choice':
            label_answer = record['answer']
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                choices=answers_list,
                target=label_answer,
                subset_key=f'level {record["level"]}',
                metadata=metadata
            )
        elif question_type == 'free_form':
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=record['answer'],
                subset_key=f'level {record["level"]}',
                metadata=metadata
            )
        else:
            raise ValueError(f'Unexpected question_type: {question_type}')

    @staticmethod
    def create_content_and_answers_list(record: Dict[str, Any], question_type) -> tuple[List[Content], List[str]]:
        """
            Create a list of content elements and a list of answers from a record.

            Args:
                record (dict): The record containing question, images, and options.
                question_type (str): The type of this question


            Returns:
                tuple: A tuple containing:
                    - content_list (list): A list of content elements (text and images).
                    - answers_list (list): A list of possible answers (for multiple-choice questions).
        """
        question: str = record['question']
        if question_type == 'multi_choice':
            answers_list = record['options']
            input_text = prompt(question=question, choices=answers_list, template=MULT_CHOICE_PROMPT)
            content_list: List[Content] = [ContentText(text=input_text)]
        else:
            answers_list: List[str] = []
            content_list: List[Content] = [ContentText(text=OPEN_PROMPT.format(question=question))]
        image = record['decoded_image']
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpg', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        return content_list, answers_list

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
