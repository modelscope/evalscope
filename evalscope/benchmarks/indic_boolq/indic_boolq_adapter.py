# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# `sarvamai/boolq-indic` ships every language in a single "default" config, differentiated by a
# `language` column, rather than one HF config per language (unlike arc_indic/triviaqa_indic/milu).
SUBSET_LIST = ['bn', 'en', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

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
        name='indic_boolq',
        pretty_name='BoolQ-Indic',
        tags=[Tags.READING_COMPREHENSION, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

BoolQ-Indic is a translation of the BoolQ yes/no reading-comprehension benchmark into 10 Indic
languages plus English, for evaluating multilingual passage understanding.

## Task Description

- **Task Type**: Multilingual Yes/No Reading Comprehension
- **Input**: Passage + yes/no question in one of 11 languages
- **Output**: `Yes` or `No`
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (validation split)
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`)
- All languages ship in a single dataset config; this adapter reformats by the `language` field
""",
        dataset_id='sarvamai/boolq-indic',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class IndicBoolQAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        language = record['language']
        label = int(record['label'])
        target_letter = 'A' if label == 1 else 'B'

        return Sample(
            input=f"{record['passage']}\n\nQuestion: {record['question']}?",
            choices=['Yes', 'No'],
            target=target_letter,
            subset_key=language,
            metadata={'language': LANGUAGE_NAMES.get(language, language)},
        )
