# flake8: noqa: E501
import re
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, Content, ContentImage, ContentText
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.io_utils import bytes_to_base64
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

logger = get_logger()

MULT_CHOICE_PROMPT = MultipleChoiceTemplate.SINGLE_ANSWER_COT

SUBSET_LIST = ['chart', 'web', 'map']


@register_benchmark(
    BenchmarkMeta(
        name='seed_bench_2_plus',
        pretty_name='SEED-Bench-2-Plus',
        dataset_id='evalscope/SEED-Bench-2-Plus',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_MODAL],
        description="""
## Overview

SEED-Bench-2-Plus is a large-scale benchmark designed to evaluate Multimodal Large Language Models (MLLMs) on text-rich visual understanding tasks. It contains 2.3K multiple-choice questions with precise human annotations across real-world scenarios.

## Task Description

- **Task Type**: Text-Rich Visual Question Answering
- **Input**: Image containing text-rich content + multiple-choice question
- **Output**: Correct answer choice letter (A/B/C/D)
- **Domains**: Charts, maps, web interfaces

## Key Features

- Focuses on text-rich visual scenarios common in real applications
- Three broad categories: **Charts**, **Maps**, and **Webs**
- Human-annotated questions with high quality
- Tests understanding of complex visual layouts with text
- Multiple difficulty levels within each category

## Evaluation Notes

- Default evaluation uses the **test** split
- Available subsets: `chart`, `web`, `map`
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Rich metadata including data source, type, and difficulty level
""",
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MULT_CHOICE_PROMPT,
    )
)
class SeedBench2PlusAdapter(VisionLanguageAdapter, MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        answers_list = [record['choice_A'], record['choice_B'], record['choice_C'], record['choice_D']]
        input_text = prompt(question=question, choices=answers_list, template=self.prompt_template)
        content_list: List[Content] = [ContentText(text=input_text)]
        image = record['image']
        if image:
            image_base64 = bytes_to_base64(image['bytes'], format='png', add_header=True)
            content_list.append(ContentImage(image=image_base64))
        metadata = {
            'data_id': record['data_id'],
            'question_id': record['question_id'],
            'question_image_subtype': record['question_image_subtype'],
            'data_source': record['data_source'],
            'data_type': record['data_type'],
            'level': record['level'],
            'subpart': record['subpart'],
            'version': record['version'],
        }
        label_answer = record['answer']
        return Sample(
            input=[ChatMessageUser(content=content_list)],
            choices=answers_list,
            target=label_answer,
            subset_key=record['question_image_type'],
            metadata=metadata,
        )
