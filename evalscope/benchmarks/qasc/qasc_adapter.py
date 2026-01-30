from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

QASC (Question Answering via Sentence Composition) is a question-answering dataset with a focus on multi-hop sentence composition. It consists of 9,980 8-way multiple-choice questions about grade school science, requiring models to combine multiple facts to arrive at the correct answer.

## Task Description

- **Task Type**: Multi-hop Science Question Answering (Multiple-Choice)
- **Input**: Science question with 8 answer choices
- **Output**: Correct answer letter
- **Focus**: Sentence composition and multi-hop reasoning

## Key Features

- 9,980 grade school science questions
- 8-way multiple-choice format
- Requires composing two facts to answer
- Tests multi-hop reasoning over scientific knowledge
- Annotated with supporting facts for each question

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on validation split
- Simple accuracy metric
- Useful for evaluating compositional reasoning
"""


@register_benchmark(
    BenchmarkMeta(
        name='qasc',
        pretty_name='QASC',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/qasc',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class QASCAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
