# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Type, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import DataLoader, Dataset, Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# 10 Indic languages, each in native script and a "roman" (Latin transliteration) variant, plus English
# (which has no roman variant since it is already in Latin script).
NATIVE_LANGS = ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']
SUBSET_LIST = ['en'] + NATIVE_LANGS + [f'{lang}_roman' for lang in NATIVE_LANGS]

LANGUAGE_NAMES = {
    'bn': 'Bengali',
    'en': 'English',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'or': 'Odia',
    'pa': 'Punjabi',
    'ta': 'Tamil',
    'te': 'Telugu',
}


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_indic',
        evaluation_version='v1.0',
        pretty_name='MMLU-Indic',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

MMLU-Indic is Sarvam AI's translation of the MMLU (Massive Multitask Language Understanding) benchmark into
10 Indic languages, each in native script and a romanized (Latin transliteration) variant, plus the original
English. It is the fourth of Sarvam's translated academic benchmarks, alongside `arc_indic`, `triviaqa_indic`
and `indic_boolq`.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: Question with answer choices in one of 21 language/script variants
- **Output**: Correct answer letter (A-D)
- **Domain**: The 57 MMLU subjects (STEM, humanities, social sciences and others)

## Key Features

- 21 subsets: English, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu
  in native script, and the 10 Indic languages again as `<lang>_roman`
- Native-script subsets have the full MMLU test set (14,042 questions) and its 285-question dev set as the
  `validation` split; the romanized subsets have 13,096-14,005 test questions and no `validation` split
- The dataset does not keep MMLU's subject labels, so results are reported per language only
- Four English test rows list three choices while their answer index points to a missing fourth choice; they are
  skipped at load time. Six other English rows have three valid choices and are kept as three-option questions

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on the `test` split, with accuracy as the metric
- Few-shot examples come from the `validation` split, available for English and the 10 native-script Indic
  subsets. The `_roman` subsets have no `validation` split and reject few-shot evaluation
- Sarvam reports zero-shot results with an "Indic Average": the mean of the 10 native-script Indic subsets,
  excluding English and the romanized subsets. This is a reporting convention for comparison with Sarvam,
  not an additional aggregate computed by this adapter
- Use `subset_list` to evaluate specific languages or scripts (e.g., `['hi', 'hi_roman']`)
""",
        dataset_id='sarvamai/mmlu-indic',
        hf_dataset_id='sarvamai/mmlu-indic',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class MMLUIndicAdapter(MultiChoiceAdapter):
    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        choices = record['choices']
        answer = record['answer']
        if not 0 <= answer < len(choices):
            # A few English rows lost an option upstream and the answer index points past the remaining choices.
            return []

        subset = self.current_subset_name
        base_lang, _, script = subset.partition('_')
        language = LANGUAGE_NAMES.get(base_lang, base_lang)
        if script == 'roman':
            language += ' (Romanized)'

        return Sample(
            input=record['question'],
            choices=choices,
            target=chr(ord('A') + answer),
            metadata={'language': language},
        )

    def load_fewshot_subset(self, subset: str, data_loader: Type[DataLoader]) -> Dataset:
        if subset.endswith('_roman'):
            raise ValueError(
                f'mmlu_indic subset `{subset}` has no `validation` split to draw few-shot examples from. '
                'Use few_shot_num=0, or evaluate the native-script subset instead.'
            )
        return super().load_fewshot_subset(subset, data_loader)
