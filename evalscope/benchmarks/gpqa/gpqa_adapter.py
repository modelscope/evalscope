# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import FEW_SHOT_TEMPLATE, MultipleChoiceTemplate

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='gpqa_diamond',
        pretty_name='GPQA-Diamond',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

GPQA (Graduate-Level Google-Proof Q&A) Diamond is a challenging benchmark of 198 multiple-choice questions written by domain experts in biology, physics, and chemistry. The questions are designed to be extremely difficult, requiring PhD-level expertise to answer correctly.

## Task Description

- **Task Type**: Expert-Level Multiple-Choice Q&A
- **Input**: Graduate-level science question with 4 choices
- **Output**: Single correct answer letter (A, B, C, or D)
- **Domains**: Biology, Physics, Chemistry

## Key Features

- 198 questions written and validated by domain PhD experts
- Questions are "Google-proof" - cannot be easily looked up
- Designed to test deep domain knowledge and reasoning
- Diamond subset represents the highest quality questions
- Average human expert accuracy ~65%, non-expert ~34%

## Evaluation Notes

- Default configuration uses **0-shot** or **5-shot** evaluation
- Supports Chain-of-Thought (CoT) prompting for improved reasoning
- Answer choices are randomly shuffled during evaluation
- Only uses train split (validation set is private)
- Challenging benchmark for measuring expert-level reasoning
""",
        dataset_id='AI-ModelScope/gpqa_diamond',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # only have train split
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class GPQAAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.few_shot_num > 0 and self.few_shot_num != 5:
            logger.warning(
                f'Only support few_shot_num 0 or 5 for {self.dataset_id}, but got {self.few_shot_num}. Use 5-shot by default.'  # noqa: E501
            )
            self.few_shot_num = 5

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        # Process the input to create shuffled choices and correct answer
        processed_data = self._process_input(record)

        return Sample(
            input=record['Question'],
            choices=processed_data['choices'],
            target=processed_data['answer'],
            subset_key=record.get('subset', ''),
            metadata={
                'correct_answer':
                record['Correct Answer'],
                'incorrect_answers':
                [record['Incorrect Answer 1'], record['Incorrect Answer 2'], record['Incorrect Answer 3']],
            },
        )

    def format_fewshot_template(self, fewshot, sample):
        from .prompt import FEW_SHOT_SAMPLES

        return FEW_SHOT_TEMPLATE.format(fewshot=FEW_SHOT_SAMPLES, ) + self.format_prompt_template(sample)

    def _process_input(self, input_d: dict) -> dict:
        """Process input to shuffle choices and determine correct answer letter."""

        def preprocess(text):
            if text is None:
                return ' '
            text = text.strip()
            text = text.replace(' [title]', '. ')
            text = re.sub('\\[.*?\\]', '', text)
            text = text.replace('  ', ' ')
            return text

        choices = [
            preprocess(input_d['Incorrect Answer 1']),
            preprocess(input_d['Incorrect Answer 2']),
            preprocess(input_d['Incorrect Answer 3']),
            preprocess(input_d['Correct Answer']),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(input_d['Correct Answer']))

        return {
            'choices': choices,
            'answer': f'{chr(65 + correct_answer_index)}',
        }
