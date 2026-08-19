# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='hellaswag_hi',
        pretty_name='HellaSwag-Hindi',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

HellaSwag-Hindi is a Hindi translation of the HellaSwag commonsense sentence-completion benchmark's
full validation set. The context stem stays in English; the 4 candidate continuations are translated
into Hindi, so the model must connect an English scenario to its most plausible Hindi-phrased ending.
Sourced from `ai4bharat/hellaswag-hi`, the same dataset used by lighteval's `community_hellaswag_hin`
tasks.

## Task Description

- **Task Type**: Commonsense Sentence Completion (mixed-language)
- **Input**: An English context sentence with 4 Hindi-language candidate continuations
- **Output**: Correct answer letter
- **Coverage**: Full HellaSwag validation set (10,042 examples)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (validation split, the only labeled split
  available — HellaSwag's `test` split ships without gold labels)
- Not mirrored on ModelScope; set `dataset_hub: huggingface` in `TaskConfig` to load it
""",
        dataset_id='ai4bharat/hellaswag-hi',
        metric_list=['acc'],
        subset_list=['hi'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class HellaSwagHiAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        target_letter = chr(ord('A') + int(record['label']))

        return Sample(
            input=record['ctx'],
            choices=record['endings'],
            target=target_letter,
            metadata={
                'activity_label': record.get('activity_label', ''),
            },
        )
