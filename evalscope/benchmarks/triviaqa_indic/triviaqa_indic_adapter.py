# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

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
        name='triviaqa_indic',
        pretty_name='TriviaQA-Indic-MCQ',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

TriviaQA-Indic-MCQ reformats TriviaQA trivia questions as 4-way multiple-choice questions, translated
into 10 Indic languages plus English, for evaluating multilingual world-knowledge recall.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Trivia Question Answering
- **Input**: Trivia question with 4 answer choices in one of 11 languages
- **Output**: Correct answer letter
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (validation split, the only split available)
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`), or `limit` to cap sample
  count — the full default run is ~18k samples per language across all 11 languages (~198k total)
""",
        dataset_id='sarvamai/trivia-qa-indic-mcq',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class TriviaQAIndicAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = record['choices']
        # answer is a 0-based index into choices
        target_letter = chr(ord('A') + int(record['answer']))

        return Sample(
            input=record['question'].strip(),
            choices=choices,
            target=target_letter,
            metadata={'language': LANGUAGE_NAMES.get(self.current_subset_name, self.current_subset_name)},
        )
