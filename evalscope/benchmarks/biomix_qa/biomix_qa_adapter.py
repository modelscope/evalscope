from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate

DESCRIPTION = """
## Overview

BiomixQA is a curated biomedical question-answering dataset designed to evaluate AI models on biomedical knowledge and reasoning. It has been utilized to validate the Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) framework across different LLMs.

## Task Description

- **Task Type**: Biomedical Multiple-Choice Question Answering
- **Input**: Biomedical question with multiple answer choices
- **Output**: Correct answer letter
- **Domain**: Biomedical sciences, healthcare, life sciences

## Key Features

- Curated biomedical questions from diverse sources
- Tests medical and biological knowledge comprehension
- Validates RAG framework effectiveness for biomedical domain
- Multiple-choice format for standardized evaluation
- Useful for evaluating healthcare AI systems

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Simple accuracy metric for performance measurement
- Evaluates on test split
- No few-shot examples provided
"""  # noqa: E501


@register_benchmark(
    BenchmarkMeta(
        name='biomix_qa',
        pretty_name='BioMixQA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.MEDICAL],
        description=DESCRIPTION.strip(),
        dataset_id='extraordinarylab/biomix-qa',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER,
    )
)
class BioMixQAAdapter(MultiChoiceAdapter):

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={},
        )
