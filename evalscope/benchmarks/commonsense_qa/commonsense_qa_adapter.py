from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

CommonsenseQA is a benchmark for evaluating AI models' ability to answer questions that require commonsense reasoning about the world. Questions are designed to require background knowledge not explicitly stated in the question.

## Task Description

- **Task Type**: Multiple-Choice Commonsense Reasoning
- **Input**: Question requiring commonsense knowledge with 5 choices
- **Output**: Correct answer letter (A-E)
- **Focus**: World knowledge and commonsense inference

## Key Features

- Questions generated from ConceptNet knowledge graph
- Requires different types of commonsense knowledge
- 5 answer choices per question
- Tests reasoning about everyday concepts and relationships
- Human-validated questions

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on validation split
- Simple accuracy metric
"""


@register_benchmark(
    BenchmarkMeta(
        name='commonsense_qa',
        pretty_name='CommonsenseQA',
        tags=[Tags.REASONING, Tags.COMMONSENSE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/commonsense-qa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class CommonsenseQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
