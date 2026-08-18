# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

# Dataset config name per language (matches ai4bharat/MILU's `configs` exactly)
SUBSET_LIST = [
    'English',
    'Bengali',
    'Gujarati',
    'Hindi',
    'Kannada',
    'Malayalam',
    'Marathi',
    'Odia',
    'Punjabi',
    'Tamil',
    'Telugu',
]

OPTION_KEYS = ['option1', 'option2', 'option3', 'option4']


@register_benchmark(
    BenchmarkMeta(
        name='milu',
        pretty_name='MILU',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MULTI_LINGUAL],
        description="""
## Overview

MILU (Multi-task Indic Language Understanding Benchmark) is a comprehensive evaluation dataset for
assessing LLM performance across 11 Indic languages. It spans 8 domains and 41 subjects, combining
translated general-knowledge questions with culturally specific Indian content.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: Question with four answer choices in one of 11 languages
- **Output**: Single correct answer letter
- **Languages**: English, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Key Features

- 8 domains / 41 subjects, including India-specific culture, history, and current affairs
- Native-language questions rather than machine-translated MMLU
- Each language is a separate dataset config, loaded independently

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split)
- Use `subset_list` to evaluate specific languages (e.g., `['Hindi', 'Tamil']`), or `limit` to cap
  sample count — evaluating all 11 languages' full test splits is a large run
- Set `few_shot_num` > 0 to enable few-shot prompting; examples are drawn from the `validation` split
- Loads from ModelScope by default (evalscope's default `dataset_hub`), where this dataset is public
  and needs no token. If you explicitly set `dataset_hub` to `huggingface`, note that
  `ai4bharat/MILU` is gated there — accept the dataset terms on huggingface.co and set `HF_TOKEN`
  (or run `huggingface-cli login`) first
""",
        dataset_id='ai4bharat/MILU',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split='validation',
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class MILUAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        choices = [record[key] for key in OPTION_KEYS]

        # target is like "option3" -> convert to a 0-based index -> letter
        target_key = record['target']
        target_index = OPTION_KEYS.index(target_key)
        target_letter = chr(ord('A') + target_index)

        return Sample(
            input=record['question'],
            choices=choices,
            target=target_letter,
            metadata={'language': self.current_subset_name},
        )
