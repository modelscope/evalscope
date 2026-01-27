import ast
from typing import Any

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='musr',
        pretty_name='MuSR',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

MuSR (Multistep Soft Reasoning) is a benchmark for evaluating complex reasoning abilities through narrative-based problems. It includes murder mysteries, object placements, and team allocation scenarios requiring multi-step inference.

## Task Description

- **Task Type**: Complex Reasoning (Multiple-Choice)
- **Input**: Narrative scenario with question and answer choices
- **Output**: Correct answer letter (A-F)
- **Domains**: Murder mysteries, object tracking, team allocation

## Key Features

- Narrative-based reasoning problems
- Requires multi-step logical inference
- Three distinct reasoning domains
- Tests constraint satisfaction and deduction
- Longer context requiring careful reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting
- Three subsets: `murder_mysteries`, `object_placements`, `team_allocation`
- Simple accuracy metric
- Challenging benchmark requiring careful reading
""",
        dataset_id='AI-ModelScope/MuSR',
        metric_list=['acc'],
        subset_list=['murder_mysteries', 'object_placements', 'team_allocation'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MuSRAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.split_as_subset = True

    def record_to_sample(self, record) -> Sample:
        choices = ast.literal_eval(record['choices'])
        choice_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        target_letter = choice_letters[record['answer_index']]

        return Sample(
            input=f"{record['narrative']}\n\n{record['question']}",
            choices=choices,
            target=target_letter,
        )
