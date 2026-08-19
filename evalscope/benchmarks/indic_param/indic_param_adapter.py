# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# `subject` values as they appear in the dataset (some long-tail language names carry irregular
# capitalization/spacing, e.g. "Sanskrit Mix", "Gujarati_surya" - preserved verbatim for subset_key
# matching).
SUBSET_LIST = [
    'Bodo',
    'Dogri',
    'Gujarati_surya',
    'Konkani',
    'Maithili',
    'Marathi',
    'Nepali',
    'Oriya',
    'Rajasthani',
    'Sanskrit',
    'Sanskrit Mix',
    'Santali',
]

OPTION_KEYS = ['option_a', 'option_b', 'option_c', 'option_d']


@register_benchmark(
    BenchmarkMeta(
        name='indic_param',
        pretty_name='IndicParam',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

IndicParam is a graduate-level benchmark evaluating LLM understanding of low- and extremely
low-resource Indic languages. All 13,207 multiple-choice questions are sourced from official UGC-NET
language question papers and answer keys, presented in each language's native script (or code-mixed
form for Sanskrit-English).

## Task Description

- **Task Type**: Graduate-Level Multiple-Choice Question Answering
- **Input**: A UGC-NET exam question with 4 answer choices, in a low-resource Indic language
- **Output**: Correct answer letter
- **Languages**: Bodo, Dogri, Gujarati (Surya script), Konkani, Maithili, Marathi, Nepali, Oriya,
  Rajasthani, Sanskrit, Sanskrit-English code-mixed, Santali

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate specific languages
- All languages ship in a single dataset config, differentiated by the `subject` field; this adapter
  reformats by that field
""",
        dataset_id='bharatgenai/IndicParam',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class IndicParamAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        subject = record['subject']
        choices = [record[key] for key in OPTION_KEYS]
        target_letter = record['correct_answer'].strip().upper()

        return Sample(
            input=record['question_text'],
            choices=choices,
            target=target_letter,
            subset_key=subject,
            metadata={
                'subject': subject,
                'exam_name': record.get('exam_name', ''),
            },
        )
