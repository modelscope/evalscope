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

OPEN_PROMPT = """
Solve the following problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
"""

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

SUBSET_LIST = [
    'arithmetic', 
    'metric geometry - length', 
    'counting', 
    'logic', 
    'graph theory', 
    'solid geometry', 
    'transformation geometry', 
    'combinatorial geometry', 
    'topology', 
    'metric geometry - area', 
    'analytic geometry', 
    'descriptive geometry', 
    'combinatorics', 
    'algebra', 
    'metric geometry - angle', 
    'statistics']


@register_benchmark(
    BenchmarkMeta(
        name='math_vision',
        pretty_name='MathVision',
        dataset_id='evalscope/MathVision',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description=
        'The MATH-Vision (MATH-V) dataset, a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions.',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
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
        if question_type == 'multi_choice':
            label_answer = record['answer']
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                choices=answers_list,
                target=label_answer,
                subset_key = record['subject'],
                metadata = {
                    'id': record['id'],
                    'image': record['image'],
                    'solution': record['solution'],
                    'level': record['level'],
                    'question_type': question_type
                }
            )
        elif question_type == 'free_form':
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=record['answer'],
                metadata={
                    'id': record['id'],
                    'image': record['image'],
                    'solution': record['solution'],
                    'level': record['level'],
                    'question_type': question_type
                }
            )
        else:
            raise ValueError(f"Unexpected question_type: {question_type}")
    
    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        question_type = task_state.metadata['question_type']
        if question_type == 'multi_choice':
            answers = parse_answers(task_state)
            return ''.join(sorted(list(answers)))
        elif question_type == 'free_form':
            pattern = r'ANSWER:\s*(.*)'
            match = re.search(pattern, prediction)
            if match:
                return match.group(1).strip()
            return ''
        else:
            raise ValueError(f'Unsupported question type: {question_type}')

    
    @staticmethod
    def create_content_and_answers_list(record: dict[str, Any], question_type) -> tuple[list[Content], list[str]]:
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
        
        if question_type == 'multi_choice':
            answers_list = record['options']
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
