# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# HF dataset config name per language
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
        name='arc_indic',
        pretty_name='ARC-Challenge-Indic',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

ARC-Challenge-Indic is a translation of the AI2 Reasoning Challenge (ARC-Challenge) science
question-answering benchmark into 10 Indic languages, plus the original English set, for evaluating
multilingual scientific reasoning.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Science Question Answering
- **Input**: Science question with answer choices in one of 11 languages
- **Output**: Correct answer letter
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split)
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`)
- Same underlying science-exam questions as `arc` (Challenge split), machine/human translated per language
""",
        dataset_id='sarvamai/arc-challenge-indic',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class ARCIndicAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choice_texts = record['choices']['text']
        answer_key = record['answerKey']
        if answer_key.isdigit():
            answer_key = chr(ord('A') + int(answer_key) - 1)

        return Sample(
            input=record['question'],
            choices=choice_texts,
            target=answer_key,
            metadata={
                'id': record.get('id', ''),
                'language': LANGUAGE_NAMES.get(self.current_subset_name, self.current_subset_name),
            },
        )
