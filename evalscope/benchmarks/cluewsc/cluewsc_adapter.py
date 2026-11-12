# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

LABEL_TO_TARGET = {'true': 'A', 'false': 'B'}

# Verbatim from OpenCompass FewCLUE_cluewsc_gen_c68933, including its mismatched quotes, so scores stay comparable.
QUESTION_TEMPLATE = '{text}\n此处，“{span2}”是否指代“{span1}“？'
PROMPT_TEMPLATE = '{question}\nA. 是\nB. 否\n请从”A“，”B“中进行选择。\n答：'


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
        prompt_template=PROMPT_TEMPLATE,
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
- Reproduces the OpenCompass `FewCLUE_cluewsc_gen_c68933` prompt and answer extraction, so scores are directly comparable

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on `test_public`; set `eval_split='dev_few_all'` to score the dev split.
- Set `few_shot_num` to draw demonstrations from `train_few_all`.
- The first uppercase letter in the response is taken as the answer, as in OpenCompass `first_capital_postprocess`; `acc` is the share of items answered correctly.
""",
    )
)
class CLUEWSCAdapter(MultiChoiceAdapter):
    """Score CLUEWSC as A/B single choice with the OpenCompass prompt and answer extraction."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        target = record['target']
        return Sample(
            input=QUESTION_TEMPLATE.format(text=record['text'], span2=target['span2_text'], span1=target['span1_text']),
            choices=['是', '否'],
            target=LABEL_TO_TARGET[record['label']],
            metadata={'idx': record['idx']},
        )

    def format_prompt_template(self, sample: Sample) -> str:
        return self.prompt_template.format(question=sample.input)

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        return f'{fewshot}\n\n{self.format_prompt_template(sample)}'

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'{self.format_prompt_template(sample)}{sample.target}'

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return next((char for char in prediction if char.isupper()), '')
