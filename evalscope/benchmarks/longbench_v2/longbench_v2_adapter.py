# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import Choices
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import answer_options, format_letter_choices

logger = get_logger()

LONGBENCH_V2_TEMPLATE = r"""Please read the following text and answer the questions below.

<text>
{document}
</text>

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}""".strip()


@register_benchmark(
    BenchmarkMeta(
        name='longbench_v2',
        pretty_name='LongBench-v2',
        tags=[Tags.READING_COMPREHENSION, Tags.MULTIPLE_CHOICE, Tags.LONG_CONTEXT],
        description="""
## Overview

LongBench v2 is a challenging benchmark for evaluating long-context understanding of large language models. It covers a wide variety of real-world tasks that require reading and comprehending long documents (ranging from a few thousand to over 2 million tokens), spanning multiple domains such as single-document QA, multi-document QA, long in-context learning, long-structured data understanding, and code repository understanding.

## Task Description

- **Task Type**: Long-Context Multiple-Choice Question Answering
- **Input**: Long document context + multiple-choice question with four answer choices (A, B, C, D)
- **Output**: Single correct answer letter
- **Domains**: Single-Doc QA, Multi-Doc QA, Long In-Context Learning, Long Structured Data Understanding, Code Repo Understanding
- **Difficulty**: Easy / Hard
- **Length**: Short / Medium / Long

## Key Features

- 503 high-quality questions requiring genuine long-document understanding
- Context lengths ranging from a few thousand tokens to over 2 million tokens
- Questions are bilingual (English and Chinese)
- Designed to require careful reading; correct answers cannot be guessed without reading the document
- Covers diverse real-world application scenarios

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (train split used as test set)
- Primary metric: **Accuracy** (exact match on letter choice)
- All four answer choices are required; no random shuffling needed
- Samples are split into **3 subsets by context length**: `short`, `medium`, `long`
- Use `subset_list` to evaluate specific length subsets (e.g., `['short', 'medium']`)
""",
        dataset_id='ZhipuAI/LongBench-v2',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        subset_list=['short', 'medium', 'long'],
        prompt_template=LONGBENCH_V2_TEMPLATE,
    )
)
class LongBenchV2Adapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True  # Split samples by 'length' field into short/medium/long subsets

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [
            record['choice_A'],
            record['choice_B'],
            record['choice_C'],
            record['choice_D'],
        ]
        return Sample(
            input=record['question'],
            choices=choices,
            target=record['answer'],  # already a letter: 'A', 'B', 'C', or 'D'
            subset_key=record.get('length', 'short'),  # Used by reformat_subset to split into short/medium/long
            metadata={
                'domain': record.get('domain', ''),
                'sub_domain': record.get('sub_domain', ''),
                'difficulty': record.get('difficulty', ''),
                'length': record.get('length', ''),
                'context': record.get('context', ''),
                '_id': record.get('_id', ''),
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        context = sample.metadata.pop('context', '') if sample.metadata else ''
        choices = Choices(sample.choices)
        choices_text = answer_options(choices)
        letters = format_letter_choices(choices)

        return self.prompt_template.format(
            document=context,
            question=sample.input,
            choices=choices_text,
            letters=letters,
        )
