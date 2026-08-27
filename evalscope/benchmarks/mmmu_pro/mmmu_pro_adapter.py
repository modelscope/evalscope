import ast
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate, answer_character, prompt

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
        description="""
## Overview

MMMU-PRO is an enhanced multimodal benchmark designed to rigorously assess the genuine understanding capabilities of advanced AI models across multiple modalities. It builds upon the original MMMU benchmark with key improvements that make evaluation more challenging and realistic.

## Task Description

- **Task Type**: Multimodal Academic Question Answering
- **Input**: Images (up to 7) + multiple-choice question
- **Output**: Correct answer choice letter
- **Domains**: 30 academic subjects across STEM, humanities, and social sciences

## Key Features

- Enhanced version of MMMU with more rigorous evaluation
- Covers 30 subjects: Accounting, Biology, Chemistry, Computer Science, Economics, Physics, etc.
- Multiple dataset formats available:
  - `standard (4 options)`: Traditional 4-choice format
  - `standard (10 options)`: Extended 10-choice format for harder evaluation
  - `vision`: Questions embedded in images
- Tests genuine multimodal understanding, not just text shortcuts

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Dataset format can be configured via `dataset_format` parameter
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Rich metadata includes topic difficulty and subject information
""",  # noqa: E501
        dataset_id='AI-ModelScope/MMMU_Pro',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
        evaluation_version='v1.1',
        extra_params={
            'dataset_format': {
                'type': 'str',
                'description': f'Dataset format variant. Choices: {DATASET_FORMATS}.',
                'value': 'standard (4 options)',
                'choices': DATASET_FORMATS,
            }
        },
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
            'subject': record.get('subject'),
        }

        answers_list: List[str] = ast.literal_eval(record['options'])

        if self.dataset_format == 'vision':
            letters = ','.join(answer_character(i) for i in range(len(answers_list)))
            input_text = VISION_PROMPT.format(letters=letters)
            content_list: List[Content] = [ContentText(text=input_text)]

            image = record.get('image')
            if image:
                normalized_image = self._normalize_media_value(image, media_type='image')
                content_list.append(self._content_image_from_value(normalized_image))
        else:
            image_map = self._extract_media(record, 'image')

            # Build prompt text
            input_text = prompt(question=record['question'], choices=answers_list, template=MULT_CHOICE_PROMPT)

            # Parse and replace image placeholders
            content_list = self._parse_text_with_media(input_text, image_map=image_map)

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=record['answer'],
            subset_key=record['subject'],
            metadata=metadata,
        )
