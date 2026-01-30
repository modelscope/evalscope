from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

SciQ is a crowdsourced science exam question dataset covering Physics, Chemistry, Biology, and other scientific domains. Most questions include supporting evidence paragraphs.

## Task Description

- **Task Type**: Science Question Answering (Multiple-Choice)
- **Input**: Science question with 4 answer choices
- **Output**: Correct answer letter (A, B, C, or D)
- **Domains**: Physics, Chemistry, Biology, Earth Science, etc.

## Key Features

- Crowdsourced science exam questions
- Multiple scientific domains covered
- Supporting evidence paragraphs available
- Tests scientific knowledge and reasoning
- Suitable for science comprehension evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses simple multiple-choice prompting
- Evaluates on test split
- Simple accuracy metric
"""


@register_benchmark(
    BenchmarkMeta(
        name='sciq',
        pretty_name='SciQ',
        tags=[Tags.READING_COMPREHENSION, Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/sciq',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class SciQAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
