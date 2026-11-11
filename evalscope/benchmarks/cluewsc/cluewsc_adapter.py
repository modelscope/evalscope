# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

LABEL_TO_TARGET = {'true': 'A', 'false': 'B'}


@register_benchmark(
    BenchmarkMeta(
        name='cluewsc',
        pretty_name='CLUEWSC',
        dataset_id='evalscope/cluewsc',
        paper_url='https://arxiv.org/abs/2107.07498',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE, Tags.CHINESE],
        subset_list=['default'],
        eval_split='test_public',
        train_split='train_few_all',
        few_shot_num=0,
        metric_list=['acc'],
        supports_choice=True,
        choice_instructions='Decide whether the pronoun in `question` refers to the named entity.',
        prompt_template=MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE,
        few_shot_prompt_template=(
            MultipleChoiceTemplate.CHINESE_FEW_SHOT_TEMPLATE + MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE
        ),
        evaluation_version='v1.0',
        description="""
## Overview

CLUEWSC is the Chinese Winograd Schema Challenge task of [FewCLUE](https://github.com/CLUEbenchmark/FewCLUE). Each item asks whether a pronoun in a sentence refers to a given noun phrase, which requires commonsense reasoning rather than surface cues.

## Task Description

- **Task Type**: Binary coreference resolution as single-choice (Chinese)
- **Input**: A Chinese sentence, a pronoun and a candidate referent
- **Output**: A (是, refers to it) or B (否, does not)
- **Domain**: Chinese commonsense reasoning

## Key Features

- Official FewCLUE splits: `test_public` (976 items, balanced labels), `dev_few_all` (159) and `train_few_all` (160)
- Same question wording as the OpenCompass `FewCLUE_cluewsc_gen` configuration, so scores are comparable

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on `test_public`; set `eval_split='dev_few_all'` to score the dev split.
- Set `few_shot_num` to draw demonstrations from `train_few_all`.
- Answers should follow the format: "答案：[LETTER]"
""",
    )
)
class CLUEWSCAdapter(MultiChoiceAdapter):
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        target = record['target']
        return Sample(
            input=f'{record["text"]}\n此处，“{target["span2_text"]}”是否指代“{target["span1_text"]}”？',
            choices=['是', '否'],
            target=LABEL_TO_TARGET[record['label']],
            metadata={'idx': record['idx']},
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'问题：{sample.input}\n选项：\nA) 是\nB) 否\n答案：{sample.target}'
