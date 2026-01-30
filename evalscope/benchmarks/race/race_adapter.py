# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='race',
        pretty_name='RACE',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

RACE (ReAding Comprehension from Examinations) is a large-scale reading comprehension benchmark collected from Chinese middle school and high school English examinations. It tests comprehensive reading comprehension abilities.

## Task Description

- **Task Type**: Reading Comprehension (Multiple-Choice)
- **Input**: Article passage with question and 4 answer choices
- **Output**: Correct answer letter (A, B, C, or D)
- **Difficulty Levels**: Middle school and High school

## Key Features

- 28,000+ passages with 100,000 questions
- Real examination questions for authentic difficulty
- Two subsets: middle (easier) and high (harder)
- Tests various comprehension skills (inference, vocabulary, main idea, etc.)
- Diverse article topics and question types

## Evaluation Notes

- Default configuration uses **3-shot** examples
- Maximum few-shot number is 3 (context length consideration)
- Uses Chain-of-Thought (CoT) prompting
- Two subsets available: `high` and `middle`
- Evaluates on test split
""",
        dataset_id='evalscope/race',
        metric_list=['acc'],
        subset_list=['high', 'middle'],
        few_shot_num=3,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class RACEAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.few_shot_num > 3:
            logger.warning(f'few_shot_num <= 3 for RACE, but got {self.few_shot_num}. Use 3-shot by default.')
            self.few_shot_num = 3

    def record_to_sample(self, record) -> Sample:
        # Format the article and question as context
        context = f"Article:\n{record['article']}\nQuestion:\n{record['question']}"

        return Sample(
            input=context,
            choices=record['options'],
            target=record['answer'],
            metadata={'example_id': record.get('example_id', 'unknown')},
        )
