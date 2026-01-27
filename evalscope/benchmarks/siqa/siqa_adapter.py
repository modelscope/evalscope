from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

SIQA (Social Interaction QA) is a benchmark for evaluating social commonsense intelligence - understanding people's actions and their social implications. Unlike benchmarks focusing on physical knowledge, SIQA tests reasoning about human behavior.

## Task Description

- **Task Type**: Social Commonsense Reasoning
- **Input**: Context about a social situation with question and 3 answer choices
- **Output**: Most socially appropriate answer (A, B, or C)
- **Focus**: Human behavior, motivations, and social implications

## Key Features

- Tests social intelligence and emotional understanding
- Questions about people's actions and their consequences
- Covers motivations, reactions, and social norms
- 33K+ crowdsourced QA pairs
- Requires reasoning about human psychology

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on validation split
- Simple accuracy metric
"""


@register_benchmark(
    BenchmarkMeta(
        name='siqa',
        pretty_name='SIQA',
        tags=[Tags.COMMONSENSE, Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/siqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class SIQAAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
