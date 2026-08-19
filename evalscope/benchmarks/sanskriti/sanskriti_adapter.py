# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Union

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

OPTION_KEYS = ['option1', 'option2', 'option3', 'option4']

SUBSET_LIST = ['association', 'country', 'gk', 'states']

# `13ari/Sanskriti` (the dataset the SANSKRITI paper itself links as its official release) ships
# all four question types in one split, distinguished by `question_type` rather than a per-subset
# config/split - hence `reformat_subset` below instead of `split_as_subset`.
QUESTION_TYPE_TO_SUBSET = {
    'Association': 'association',
    'Country Prediction': 'country',
    'General Awareness': 'gk',
    'State Prediction': 'states',
}


@register_benchmark(
    BenchmarkMeta(
        name='sanskriti',
        pretty_name='Sanskriti',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

Sanskriti is a multiple-choice trivia benchmark testing knowledge of Indian states' culture, history,
and geography, sourced from state-specific attributes (art, cuisine, festivals, etc.) with
Wikipedia-backed answers. From the SANSKRITI paper (arXiv:2506.15355); this adapter loads the
dataset the paper itself links as its official release.

## Task Description

- **Task Type**: Multiple-Choice Trivia Question Answering
- **Input**: A question about a specific Indian state's culture/geography/history, with 4 answer choices
- **Output**: Correct answer letter
- **Subsets**: `association` (state-attribute association trivia), `country` (country-level trivia),
  `gk` (general knowledge), `states` (state-identification trivia)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (the dataset's only split, named `train` upstream
  despite being evaluation data)
- Questions and choices are in English
- Not mirrored on ModelScope; set `dataset_hub: huggingface` in `TaskConfig` to load it
- The paper acknowledges some questions involve ambiguous cultural elements; a small number of rows
  (~0.6%) whose `answer` doesn't match any of the 4 listed options are skipped at load time
""",
        dataset_id='13ari/Sanskriti',
        metric_list=['acc'],
        subset_list=SUBSET_LIST,
        few_shot_num=0,
        train_split=None,
        eval_split='train',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class SanskritiAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Union[Sample, List[Sample]]:
        choices = [record[key] for key in OPTION_KEYS]
        if record['answer'] not in choices:
            # A small fraction of upstream rows have an `answer` that doesn't match any option
            # (typos or genuinely mismatched keys); skip rather than crash on choices.index().
            return []
        target_index = choices.index(record['answer'])
        target_letter = chr(ord('A') + target_index)
        subset = QUESTION_TYPE_TO_SUBSET[record['question_type']]

        return Sample(
            input=record['question'],
            choices=choices,
            target=target_letter,
            subset_key=subset,
            metadata={
                'state': record.get('state', ''),
                'attribute': record.get('attribute', ''),
            },
        )
