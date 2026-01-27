from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='winogrande',
        pretty_name='Winogrande',
        tags=[Tags.REASONING, Tags.MULTIPLE_CHOICE],
        description="""
## Overview

Winogrande is a large-scale benchmark for commonsense reasoning, specifically designed to test pronoun resolution in the Winograd Schema Challenge format. It contains 44K problems that require understanding of physical and social commonsense.

## Task Description

- **Task Type**: Pronoun Resolution / Commonsense Reasoning
- **Input**: Sentence with ambiguous pronoun and two options
- **Output**: Correct option (A or B) that resolves the pronoun
- **Format**: Binary choice between two noun phrases

## Key Features

- 44K Winograd-style pronoun resolution problems
- Adversarially filtered to reduce dataset biases
- Tests physical commonsense (object properties, actions)
- Tests social commonsense (intentions, emotions)
- Requires understanding context to resolve ambiguity

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Binary choice format (option1 vs option2)
- Answers are converted to A/B letter format
- Simple accuracy metric for evaluation
- Commonly used for commonsense reasoning assessment
""",
        dataset_id='AI-ModelScope/winogrande_val',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class WinograndeAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['sentence'],
            choices=[record['option1'], record['option2']],
            target=chr(ord('A') + int(record['answer']) - 1),  # Convert 1,2 to A,B
            metadata={'id': record.get('id', 'unknown')},
        )
