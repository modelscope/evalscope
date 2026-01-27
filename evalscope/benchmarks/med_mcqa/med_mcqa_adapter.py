from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

MedMCQA is a large-scale multiple-choice question answering dataset designed to address real-world medical entrance exam questions. It contains over 194K questions covering diverse medical topics from Indian medical entrance examinations (AIIMS, NEET-PG).

## Task Description

- **Task Type**: Medical Knowledge Multiple-Choice QA
- **Input**: Medical question with 4 answer choices
- **Output**: Correct answer letter
- **Domain**: Clinical medicine, basic sciences, healthcare

## Key Features

- Over 194,000 medical exam questions
- Real questions from AIIMS and NEET-PG exams
- 21 medical subjects covered (anatomy, pharmacology, pathology, etc.)
- Expert-verified correct answers with explanations
- Tests medical knowledge comprehension and clinical reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on validation split
- Simple accuracy metric
- Train split available for few-shot learning
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='med_mcqa',
        pretty_name='Med-MCQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/medmcqa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split='train',
        eval_split='validation',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class MedMCQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
