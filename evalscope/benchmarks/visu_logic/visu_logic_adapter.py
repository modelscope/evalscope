# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger

logger = get_logger()

MULT_CHOICE_PROMPT = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}
"""

SUBSET_LIST = [
    'Quantitative Reasoning', 'Other', 'Positional Reasoning', 'Stylistic Reasoning', 'Spatial Reasoning',
    'Attribute Reasoning'
]


@register_benchmark(
    BenchmarkMeta(
        name='visulogic',
        pretty_name='VisuLogic',
        dataset_id='evalscope/VisuLogic',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description="""
## Overview

VisuLogic is a benchmark for evaluating visual reasoning capabilities of Multimodal Large Language Models (MLLMs), independent of textual reasoning. It features carefully constructed visual reasoning tasks that are inherently difficult to articulate using language alone.

## Task Description

- **Task Type**: Visual Reasoning (Multiple-Choice)
- **Input**: Image + visual reasoning question with 4 choices
- **Output**: Answer letter (A/B/C/D)
- **Domains**: Pure visual reasoning without text-based shortcuts

## Key Features

- Six reasoning skill categories:
  - **Quantitative Reasoning**: Understanding quantity changes in images
  - **Positional Reasoning**: Understanding spatial positions
  - **Spatial Reasoning**: Understanding 3D spatial relationships
  - **Attribute Reasoning**: Understanding visual attributes
  - **Stylistic Reasoning**: Understanding visual styles
  - **Other**: Miscellaneous visual reasoning tasks
- Tests genuine visual understanding beyond language shortcuts

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting with "ANSWER: [LETTER]" format
- Results grouped by reasoning skill category
""",
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class VisuLogicAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record.get('question', '')
        content_list: List[Content] = []
        prompt_text = self.prompt_template.format(question=question).strip()
        content_list.append(ContentText(text=prompt_text))

        image = record.get('image')
        if image and isinstance(image, dict):
            image_bytes = image.get('bytes')
            if image_bytes:
                image_base64 = bytes_to_base64(image_bytes, format='png', add_header=True)
                content_list.append(ContentImage(image=image_base64))

        metadata = {
            'id': record['id'],
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=record['label'],
            choices=['A', 'B', 'C', 'D'],
            subset_key=record['tag'],
            metadata=metadata,
        )
