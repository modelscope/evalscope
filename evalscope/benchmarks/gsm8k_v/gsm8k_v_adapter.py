# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """You are an expert at solving mathematical word problems. Please solve the following problem step by step, showing your reasoning.

When providing your final answer:
- If the answer can be expressed as a whole number (integer), provide it as an integer
Problem: {question}

Please think step by step. After your reasoning, put your final answer within \\boxed{{}} with the number only.
""".lstrip()  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='gsm8k_v',
        pretty_name='GSM8K-V',
        dataset_id='evalscope/GSM8K-V',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTI_MODAL],
        description="""
## Overview

GSM8K-V is a purely visual multi-image mathematical reasoning benchmark that systematically transforms each GSM8K math word problem into its visual counterpart. It enables clean within-item comparison across modalities for multimodal math evaluation.

## Task Description

- **Task Type**: Visual Mathematical Word Problem Solving
- **Input**: Multiple images representing a math problem + question
- **Output**: Numerical answer in \\boxed{} format
- **Domains**: Visual math reasoning, scene understanding, arithmetic

## Key Features

- Visual transformation of GSM8K text problems
- Multi-image input format for complete problem representation
- Enables text-vs-visual modality comparisons
- Categorized by problem type and subcategory
- Preserves original problem difficulty levels

## Evaluation Notes

- Default evaluation uses the **train** split
- Primary metric: **Accuracy** with numeric comparison
- Answers should be in \\boxed{} format with number only
- Step-by-step reasoning encouraged before final answer
- Metadata includes original text question for comparison
""",  # noqa: E501
        subset_list=['default'],
        eval_split='train',
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class GSM8KVAdapter(VisionLanguageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['modify_scene_related_question']
        content_list: List[Content] = []
        input_text = self.prompt_template.format(question=question).strip()
        target = str(record['answer'])
        content_list.append(ContentText(text=input_text))

        images = record.get('image')
        if images:
            base64_list = json.loads(images)
            for image_value in base64_list:
                content_list.append(ContentImage(image=f'data:image/jpeg;base64,{image_value}'))

        return Sample(
            input=[ChatMessageUser(content=content_list)],
            target=target,
            metadata={
                'index': record['index'],
                'category': record['category'],
                'subcategory': record['subcategory'],
                'original_question': record['original_question']
            }
        )

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
