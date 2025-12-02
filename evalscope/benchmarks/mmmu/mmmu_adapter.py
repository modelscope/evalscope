import ast
import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, parse_answers, prompt

logger = get_logger()

SUBSET_LIST = [
    'Accounting',
    'Agriculture',
    'Architecture_and_Engineering',
    'Art',
    'Art_Theory',
    'Basic_Medical_Science',
    'Biology',
    'Chemistry',
    'Clinical_Medicine',
    'Computer_Science',
    'Design',
    'Diagnostics_and_Laboratory_Medicine',
    'Economics',
    'Electronics',
    'Energy_and_Power',
    'Finance',
    'Geography',
    'History',
    'Literature',
    'Manage',
    'Marketing',
    'Materials',
    'Math',
    'Mechanical_Engineering',
    'Music',
    'Pharmacy',
    'Physics',
    'Psychology',
    'Public_Health',
    'Sociology',
]

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

OPEN_PROMPT = """
Solve the following problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \\boxed command.

"""  # noqa: E501

MULTI_CHOICE_TYPE = 'multiple-choice'
OPEN_TYPE = 'open'


@register_benchmark(
    BenchmarkMeta(
        name='mmmu',
        pretty_name='MMMU',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description=
        'MMMU (A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI) benchmark designed to evaluate multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. These questions span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures.',  # noqa: E501
        dataset_id='AI-ModelScope/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=OPEN_PROMPT,
    )
)
class MMMUAdapter(VisionLanguageAdapter):
    """
    example1:{
        'question': '<image 1> illustrate a walk and a cycle. We can easily represent walking as?'
        'options': "['a line', 'a curve', 'a plane', 'a surface']",
        'image_1': {'bytes': b'...'},
        'question_type': 'multiple-choice',
    }
    example2:{
        'question': 'Select the correct tuning of Violin.',
        'options': "[<image 1>, <image 2>, <image 3>, <image 4>]",
        'image_1': {'bytes': b'...'},
        'image_2': {'bytes': b'...'},
        'image_3': {'bytes': b'...'},
        'image_4': {'bytes': b'...'},
        'question_type': 'multiple-choice',
    }
    example3:{
        'question': 'Each of seven students has chosen three courses from ten options, and must sit an exam for each of his or her three choices. Two students sitting the same exam must do so at the same time, but no student can sit more than one exam in the same day. The table of choices is given in <image 1>. Find the smallest number of days required to schedule the exams. Return only the number of days.',
        'options': '[]',
        'image_1': {'bytes': b'...'},
        'question_type': 'open',
    }
    """  # noqa: E501
    MAX_IMAGES: int = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question_type = record['question_type']
        content_list, answers_list = self.create_content_and_answers_list(record)

        metadata = {
            'id': record['id'],
            'question_type': record['question_type'],
            'subfield': record['subfield'],
            'explanation': record['explanation'],
            'img_type': record['img_type'],
            'topic_difficulty': record['topic_difficulty'],
        }

        if question_type == MULTI_CHOICE_TYPE:
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                choices=answers_list,
                target=record['answer'],
                metadata=metadata,
            )
        elif question_type == OPEN_TYPE:
            return Sample(
                input=[ChatMessageUser(content=content_list)],
                target=record['answer'],
                metadata=metadata,
            )
        else:
            raise ValueError(f'Unsupported question type: {question_type}')

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
            return prediction.strip()
        else:
            raise ValueError(f'Unsupported question type: {question_type}')

    def create_content_and_answers_list(self, record: Dict[str, Any]) -> tuple[List[Content], List[str]]:
        """
        Create a list of content elements and a list of answers from a record.
        Images are inserted at their <image x> placeholder positions in the text.

        Args:
            record (dict): The record containing question, images, and options.

        Returns:
            tuple: A tuple containing:
                - content_list (list): A list of content elements (text and images).
                - answers_list (list): A list of possible answers (for multiple-choice questions).
        """
        question_type = record['question_type']

        # Prepare image map
        image_map: Dict[int, str] = {}
        for i in range(MMMUAdapter.MAX_IMAGES):
            image = record.get(f'image_{i+1}')
            if image:
                image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
                image_map[i + 1] = image_base64

        if question_type == MULTI_CHOICE_TYPE:
            answers_list: List[str] = ast.literal_eval(record['options'])

            # Build prompt text
            full_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)

            # Parse and replace image placeholders
            content_list = self._parse_text_with_images(full_text, image_map)

        else:  # OPEN_TYPE
            answers_list: List[str] = []
            full_text = OPEN_PROMPT.format(question=record['question'])
            content_list = self._parse_text_with_images(full_text, image_map)

        return content_list, answers_list
