# Copyright (c) Alibaba, Inc. and its affiliates.

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='arc',
        pretty_name='ARC',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

ARC (AI2 Reasoning Challenge) is a benchmark designed to evaluate science question answering capabilities of AI models. It consists of multiple-choice science questions from grade 3 to grade 9, divided into an Easy set and a Challenge set based on difficulty.

## Task Description

- **Task Type**: Multiple-Choice Science Question Answering
- **Input**: Science question with 3-5 answer choices
- **Output**: Correct answer letter (A, B, C, D, or E)
- **Difficulty Levels**: ARC-Easy and ARC-Challenge

## Key Features

- 7,787 science questions from standardized tests (grades 3-9)
- ARC-Easy: Questions answerable by retrieval or word co-occurrence
- ARC-Challenge: Questions requiring deeper reasoning
- Questions cover physics, chemistry, biology, and earth science
- Designed to test both factual knowledge and reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Two subsets available: `ARC-Easy` and `ARC-Challenge`
- Challenge set is commonly used for leaderboard comparisons
- Supports few-shot evaluation with train split examples
""",
        dataset_id='allenai/ai2_arc',
        subset_list=['ARC-Easy', 'ARC-Challenge'],
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class ARCAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        # Convert choice labels to indices (A->0, B->1, etc.)
        choice_texts = record['choices']['text']
        answer_key = record['answerKey']

        return Sample(
            input=record['question'],
            choices=choice_texts,
            target=answer_key,
            metadata={
                'id': record.get('id', ''),
            },
        )
