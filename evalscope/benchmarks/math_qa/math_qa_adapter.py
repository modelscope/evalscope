from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

MathQA is a large-scale dataset for mathematical word problem solving, gathered by annotating the AQuA-RAT dataset with fully-specified operational programs using a new representation language. It contains diverse math problems requiring multi-step reasoning.

## Task Description

- **Task Type**: Mathematical Reasoning (Multiple-Choice)
- **Input**: Math word problem with multiple answer choices
- **Output**: Correct answer with chain-of-thought reasoning
- **Difficulty**: Varied (elementary to intermediate level)

## Key Features

- Annotated with executable operational programs
- Tests quantitative reasoning and problem-solving skills
- Diverse mathematical topics and question formats
- Multiple-choice format with structured solutions
- Useful for evaluating mathematical reasoning capabilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Evaluates on test split
- Simple accuracy metric
- Reasoning steps available in metadata
"""


@register_benchmark(
    BenchmarkMeta(
        name='math_qa',
        pretty_name='MathQA',
        tags=[Tags.REASONING, Tags.MATH, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/math-qa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class MathQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={'reasoning': record['reasoning']},
        )
