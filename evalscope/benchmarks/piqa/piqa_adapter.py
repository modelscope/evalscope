from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

PIQA (Physical Interaction QA) is a benchmark for evaluating AI models' understanding of physical commonsense - how objects interact in the physical world and what happens when we manipulate them.

## Task Description

- **Task Type**: Physical Commonsense Reasoning
- **Input**: Goal/question with two possible solutions
- **Output**: More physically plausible solution (A or B)
- **Focus**: Physical world knowledge and intuitive physics

## Key Features

- Tests understanding of physical object properties
- Binary choice between plausible/implausible solutions
- Requires intuitive physics reasoning
- Covers everyday physical scenarios
- Adversarially filtered to reduce biases

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on validation split
- Simple accuracy metric
"""


@register_benchmark(
    BenchmarkMeta(
        name='piqa',
        pretty_name='PIQA',
        tags=[Tags.REASONING, Tags.COMMONSENSE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/piqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class PIQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
