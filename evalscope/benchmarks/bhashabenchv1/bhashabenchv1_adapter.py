# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# Unlike bhasha_bench_multi, each of these 4 repos puts English/Hindi in their own HF config with
# plain `question`/`option_*` fields holding that config's own language directly (no separate
# `*_translated` fields) - so this follows the standard per-language-config pattern.
SUBSET_LIST = ['English', 'Hindi']

OPTION_KEYS = ['option_a', 'option_b', 'option_c', 'option_d']

BHASHABENCHV1_DESCRIPTION_TEMPLATE = """
## Overview

BhashaBench-{domain} is the predecessor of BhashaBench-Multi's {domain_lower} domain: a domain-specific
multiple-choice benchmark evaluating LLM knowledge of {domain_desc}, covering English and Hindi.

## Task Description

- **Task Type**: Domain-Specific Multiple-Choice Question Answering
- **Input**: {article} {domain_desc} question with 4 answer choices, in English or Hindi
- **Output**: Correct answer letter
- **Languages**: English, Hindi

## Key Features

- 5,600–17,000 questions per language, covering English and Hindi only
- Predecessor of BhashaBench-Multi: same domains, narrower language coverage
- Each domain is a separate repository, with English and Hindi as separate configs

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate a single language (e.g., `['Hindi']`)
- Requires access to this gated dataset - on ModelScope (the default hub), accept the terms and
  ensure you're logged in; alternatively, set `dataset_hub` to `huggingface` and use `HF_TOKEN`
  after accepting the terms on huggingface.co
- For broader language coverage of the same domain, see `bhasha_bench_multi_{domain_lower}`
  (22 Indic languages, not gated)
"""


@register_benchmark(
    BenchmarkMeta(
        name='bhashabenchv1_ayur',
        pretty_name='BhashaBench-V1 (Ayurveda)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHABENCHV1_DESCRIPTION_TEMPLATE.format(
            domain='Ayur', domain_lower='ayur', domain_desc='Ayurvedic medicine', article='An'
        ),
        dataset_id='bharatgenai/BhashaBench-Ayur',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchV1AyurAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record[key] for key in OPTION_KEYS]
        target_letter = record['correct_answer'].strip().upper()

        return Sample(
            input=record['question'],
            choices=choices,
            target=target_letter,
            metadata={
                'language': self.current_subset_name,
                'topic': record.get('topic', ''),
            },
        )


@register_benchmark(
    BenchmarkMeta(
        name='bhashabenchv1_finance',
        pretty_name='BhashaBench-V1 (Finance)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHABENCHV1_DESCRIPTION_TEMPLATE.format(
            domain='Finance', domain_lower='finance', domain_desc='finance', article='A'
        ),
        dataset_id='bharatgenai/BhashaBench-Finance',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchV1FinanceAdapter(BhashaBenchV1AyurAdapter):
    ...


@register_benchmark(
    BenchmarkMeta(
        name='bhashabenchv1_krishi',
        pretty_name='BhashaBench-V1 (Krishi)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHABENCHV1_DESCRIPTION_TEMPLATE.format(
            domain='Krishi', domain_lower='krishi', domain_desc='agriculture (Krishi)', article='An'
        ),
        dataset_id='bharatgenai/BhashaBench-Krishi',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchV1KrishiAdapter(BhashaBenchV1AyurAdapter):
    ...


@register_benchmark(
    BenchmarkMeta(
        name='bhashabenchv1_legal',
        pretty_name='BhashaBench-V1 (Legal)',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description=BHASHABENCHV1_DESCRIPTION_TEMPLATE.format(
            domain='Legal', domain_lower='legal', domain_desc='Indian law', article='An'
        ),
        dataset_id='bharatgenai/BhashaBench-Legal',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BhashaBenchV1LegalAdapter(BhashaBenchV1AyurAdapter):
    ...
