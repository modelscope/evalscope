import ast
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, prompt

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

VISION_PROMPT = r"""
Answer the following multiple choice question in image. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

""".strip()  # noqa: E501

DATASET_FORMATS = ['standard (4 options)', 'standard (10 options)', 'vision']


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pro',
        pretty_name='MMMU-PRO',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=
        'MMMU-Pro is an enhanced multimodal benchmark designed to rigorously assess the true understanding capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU benchmark by introducing several key improvements that make it more challenging and realistic, ensuring that models are evaluated on their genuine ability to integrate and comprehend both visual and textual information.',  # noqa: E501
        dataset_id='AI-ModelScope/MMMU_Pro',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
        extra_params={
            'dataset_format': {
                'type': 'str',
                'description': f'Dataset format variant. Choices: {DATASET_FORMATS}.',
                'value': 'standard (4 options)',
                'choices': DATASET_FORMATS
            }
        }
    )
)
class MMMUPROAdapter(VisionLanguageAdapter, MultiChoiceAdapter):
    MAX_IMAGES: int = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reformat_subset = True
        self.dataset_format = self.extra_params.get('dataset_format', 'standard (4 options)')
        self.default_subset = self.dataset_format

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:

        metadata = {
            'id': record['id'],
            'explanation': record.get('explanation'),
            'img_type': record.get('img_type'),
            'topic_difficulty': record.get('topic_difficulty'),
            'subject': record.get('subject')
        }

        answers_list: List[str] = ast.literal_eval(record['options'])

        if self.dataset_format == 'vision':
            letters = ','.join(answer_character(i) for i in range(len(answers_list)))
            input_text = VISION_PROMPT.format(letters=letters)
            content_list: List[Content] = [ContentText(text=input_text)]

            image = record.get('image')
            if image:
                content_list.append(ContentImage(image=bytes_to_base64(image['bytes'], format='png', add_header=True)))
        else:
            # Prepare image map
            image_map: Dict[int, str] = {}
            for i in range(MMMUPROAdapter.MAX_IMAGES):
                image = record.get(f'image_{i+1}')
                if image:
                    image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
                    image_map[i + 1] = image_base64

            # Build prompt text
            input_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)

            # Parse and replace image placeholders
            content_list = self._parse_text_with_images(input_text, image_map)

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=record['answer'],
            subset_key=record['subject'],
            metadata=metadata,
        )
