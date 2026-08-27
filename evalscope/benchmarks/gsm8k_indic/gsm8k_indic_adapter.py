# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.""".lstrip()  # noqa: E501

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
        name='gsm8k_indic',
        pretty_name='GSM8K-Indic',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTI_LINGUAL],
        description="""
## Overview

GSM8K-Indic translates the GSM8K grade-school math word problems into 10 Indic languages, each
available in native script and a romanized (Latin transliteration) variant, plus the original English.

## Task Description

- **Task Type**: Multilingual Mathematical Word Problem Solving
- **Input**: Grade-school math word problem in one of 21 language/script variants
- **Output**: Numerical answer derived through step-by-step reasoning
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil,
  Telugu — each Indic language in both native script and a `_roman` transliterated variant

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate specific languages/scripts (e.g., `['hi', 'hi_roman']`)
- Gold answers are the original English reasoning chain's final numeric value; only the question is
  translated
""",
        dataset_id='sarvamai/gsm8k-indic',
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        metric_list=[{'acc': {'numeric': True}}],
        prompt_template=PROMPT_TEMPLATE,
    )
)
class GSM8KIndicAdapter(DefaultDataAdapter):
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        DELIM = '####'
        question = record['question']
        answer = record['answer'].split(DELIM)
        target = answer.pop().strip()

        subset = self.current_subset_name
        base_lang, _, script = subset.partition('_')
        language = LANGUAGE_NAMES.get(base_lang, base_lang)
        if script == 'roman':
            language += ' (Romanized)'

        return Sample(input=question, target=target, metadata={'language': language})

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math.parser import extract_answer

        return extract_answer(prediction)
