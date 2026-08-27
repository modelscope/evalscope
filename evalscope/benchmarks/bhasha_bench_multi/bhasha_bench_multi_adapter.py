# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# `bharatgenai/BhashaBench-Multi` puts each domain in its own HF config (BBA/BBF/BBK/BBL) and each
# language in its own HF *split* within that config (not a per-language config) - so language
# selection here goes through `split_as_subset`, and the domain is fixed per adapter via
# `default_subset`. There is no English split for this dataset (its 22 languages are all Indic).
LANGUAGE_SPLITS = [
    'Assamese',
    'Bengali',
    'Bodo',
    'Dogri',
    'Gujarati',
    'Hindi',
    'Kannada',
    'Kashmiri',
    'Konkani',
    'Maithili',
    'Malayalam',
    'Manipuri',
    'Marathi',
    'Nepali',
    'Oriya',
    'Punjabi',
    'Sanskrit',
    'Santhali',
    'Sindhi',
    'Tamil',
    'Telugu',
    'Urdu',
]

OPTION_KEYS = ['option_a_translated', 'option_b_translated', 'option_c_translated', 'option_d_translated']

BHASHA_BENCH_MULTI_DESCRIPTION_TEMPLATE = """
## Overview

BhashaBench-Multi ({domain}) is a domain-specific multiple-choice benchmark evaluating LLM knowledge
of {domain_desc} across 22 Indic languages. Each question originates in English and is machine
translated (with LLM-judged translation quality scores) into the target language; this adapter uses
the translated question/choices.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: A {domain_desc} question with 4 answer choices, in one of 22 Indic languages
- **Output**: Correct answer letter
- **Languages**: Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili,
  Malayalam, Manipuri, Marathi, Nepali, Oriya, Punjabi, Sanskrit, Santhali, Sindhi, Tamil, Telugu, Urdu

## Key Features

- ~14,963 questions per language across 22 Indic languages per domain (~330k total per domain)
- Machine-translated from English with LLM-judged translation quality scores
- 22 scheduled languages of India, all in native script; no English split
- Four domains available as separate benchmarks: Ayurveda, Finance, Krishi, Legal

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate specific languages (e.g., `['Hindi', 'Tamil']`), or `limit` to cap
  sample count — each domain is ~14,963 questions per language across 22 languages (~330k total),
  so evaluating every language's full split is a large run
- No English split exists for this dataset
"""


@register_benchmark(
    BenchmarkMeta(
        name='bhasha_bench_multi_ayur',
        pretty_name='BhashaBench-Multi (Ayurveda)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHA_BENCH_MULTI_DESCRIPTION_TEMPLATE.format(domain='Ayurveda', domain_desc='Ayurvedic medicine'),
        dataset_id='bharatgenai/BhashaBench-Multi',
        default_subset='BBA',
        metric_list=['acc'],
        subset_list=LANGUAGE_SPLITS,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchMultiAyurAdapter(MultiChoiceAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_as_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record[key] for key in OPTION_KEYS]
        target_letter = record['correct_answer'].strip().upper()

        return Sample(
            input=record['question_translated'],
            choices=choices,
            target=target_letter,
            metadata={
                'language': self.current_subset_name,
                'topic': record.get('topic', ''),
            },
        )


@register_benchmark(
    BenchmarkMeta(
        name='bhasha_bench_multi_finance',
        pretty_name='BhashaBench-Multi (Finance)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHA_BENCH_MULTI_DESCRIPTION_TEMPLATE.format(domain='Finance', domain_desc='finance'),
        dataset_id='bharatgenai/BhashaBench-Multi',
        default_subset='BBF',
        metric_list=['acc'],
        subset_list=LANGUAGE_SPLITS,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchMultiFinanceAdapter(BhashaBenchMultiAyurAdapter): ...


@register_benchmark(
    BenchmarkMeta(
        name='bhasha_bench_multi_krishi',
        pretty_name='BhashaBench-Multi (Krishi)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHA_BENCH_MULTI_DESCRIPTION_TEMPLATE.format(domain='Krishi', domain_desc='agriculture (Krishi)'),
        dataset_id='bharatgenai/BhashaBench-Multi',
        default_subset='BBK',
        metric_list=['acc'],
        subset_list=LANGUAGE_SPLITS,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchMultiKrishiAdapter(BhashaBenchMultiAyurAdapter): ...


@register_benchmark(
    BenchmarkMeta(
        name='bhasha_bench_multi_legal',
        pretty_name='BhashaBench-Multi (Legal)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHA_BENCH_MULTI_DESCRIPTION_TEMPLATE.format(domain='Legal', domain_desc='Indian law'),
        dataset_id='bharatgenai/BhashaBench-Multi',
        default_subset='BBL',
        metric_list=['acc'],
        subset_list=LANGUAGE_SPLITS,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchMultiLegalAdapter(BhashaBenchMultiAyurAdapter): ...
