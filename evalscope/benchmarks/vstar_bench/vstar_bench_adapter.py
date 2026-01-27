from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import parse_answers

# flake8: noqa

logger = get_logger()

MULT_CHOICE_PROMPT = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.

{question}
"""


@register_benchmark(
    BenchmarkMeta(
        name='vstar_bench',
        pretty_name='V*Bench',
        dataset_id='lmms-lab/vstar-bench',
        tags=[Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL, Tags.GROUNDING],
        description="""
## Overview

V*Bench is a benchmark designed for evaluating visual search capabilities within multimodal reasoning systems. It focuses on actively locating and identifying specific visual information in high-resolution images, crucial for fine-grained visual understanding.

## Task Description

- **Task Type**: Visual Search and Reasoning (Multiple-Choice)
- **Input**: High-resolution image + targeted visual query
- **Output**: Answer letter (A/B/C/D)
- **Domains**: Visual search, fine-grained recognition, visual grounding

## Key Features

- Tests targeted visual query capabilities
- Focuses on high-resolution image understanding
- Requires finding and reasoning about specific visual elements
- Questions guided by natural language instructions
- Evaluates fine-grained visual understanding in complex scenes

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting with "ANSWER: [LETTER]" format
- Metadata includes category and question ID for analysis
""",
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class VstarBenchAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question: str = record.get('text', '')
        content_list: List[Content] = []
        prompt_text = self.prompt_template.format(question=question).strip()
        content_list.append(ContentText(text=prompt_text))

        image = record.get('image')
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='jpeg', add_header=True)
            content_list.append(ContentImage(image=image_base64))

        target = record.get('label', '')

        metadata: Dict[str, Any] = {
            'category': record.get('category'),
            'question_id': record.get('question_id'),
        }

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=['A', 'B', 'C', 'D'],
            target=target,
            metadata=metadata,
        )
