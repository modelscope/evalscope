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
        description=
        'GPQA is a dataset for evaluating the reasoning ability of large language models (LLMs) on complex mathematical problems. It contains questions that require step-by-step reasoning to arrive at the correct answer.',  # noqa: E501
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
