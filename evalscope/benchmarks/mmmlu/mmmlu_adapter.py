# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import Choices
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmlu.prompt import LANGUAGE_PROMPT_MAP
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate, prompt

logger = get_logger()

# Language code to display name mapping
LANGUAGE_MAPPING = {
    'AR_XY': 'Arabic',
    'BN_BD': 'Bengali',
    'DE_DE': 'German',
    'ES_LA': 'Spanish',
    'FR_FR': 'French',
    'HI_IN': 'Hindi',
    'ID_ID': 'Indonesian',
    'IT_IT': 'Italian',
    'JA_JP': 'Japanese',
    'KO_KR': 'Korean',
    'PT_BR': 'Portuguese',
    'SW_KE': 'Swahili',
    'YO_NG': 'Yoruba',
    'ZH_CN': 'Chinese',
}

SUBSET_LIST = list(LANGUAGE_MAPPING.keys())


@register_benchmark(
    BenchmarkMeta(
        name='mmmlu',
        pretty_name='MMMLU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

MMMLU (Multilingual Massive Multitask Language Understanding) is a multilingual extension of the MMLU benchmark. It evaluates the multilingual knowledge and reasoning capabilities of language models across 14 languages, covering 57 subjects from the original MMLU benchmark.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: Question with four answer choices (A, B, C, D) in one of 14 languages
- **Output**: Single correct answer letter
- **Languages**: Arabic, Bengali, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Portuguese, Swahili, Yoruba, Chinese
- **Subjects**: 57 subjects from MMLU (STEM, Humanities, Social Sciences, Other)

## Key Features

- Multilingual translation of the full MMLU benchmark
- 14 typologically diverse languages covering major language families
- Tests cross-lingual knowledge transfer and multilingual reasoning
- Same subject coverage as original MMLU (57 subjects)
- Includes low-resource languages (e.g., Swahili, Yoruba)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split only)
- Use `subset_list` to evaluate specific languages (e.g., `['ZH_CN', 'JA_JP', 'FR_FR']`)
- Results are grouped by language subset
- Cross-lingual performance comparison supported
""",
        dataset_id='openai-mirror/MMMLU',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MMMLUAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [
            record['A'],
            record['B'],
            record['C'],
            record['D'],
        ]
        return Sample(
            input=record['Question'],
            choices=choices,
            target=record['Answer'],  # already a letter: 'A', 'B', 'C', or 'D'
            metadata={
                'subject': record['Subject'],
                'language': self.current_subset_name,
            },
        )

    def format_prompt_template(self, sample: Sample) -> str:
        language = sample.metadata.get('language') if sample.metadata else None
        template = LANGUAGE_PROMPT_MAP.get(language, self.prompt_template)
        return prompt(
            question=sample.input,
            choices=Choices(sample.choices),
            template=template,
        )
