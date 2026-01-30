from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.multi_choices import MultipleChoiceTemplate


@register_benchmark(
    BenchmarkMeta(
        name='iquiz',
        pretty_name='IQuiz',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE, Tags.CHINESE],
        description="""
## Overview

IQuiz is a Chinese benchmark for evaluating AI models on intelligence quotient (IQ) and emotional quotient (EQ) questions. It tests logical reasoning, pattern recognition, and social-emotional understanding through multiple-choice questions.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Question in Chinese with multiple choice options
- **Output**: Selected answer with explanation (Chain-of-Thought)
- **Language**: Chinese

## Key Features

- Dual evaluation of IQ and EQ capabilities
- Chinese-language cognitive assessment
- Multiple difficulty levels
- Requires explanation alongside answer selection
- Tests logical reasoning and emotional understanding

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- Subsets: **IQ** (logical reasoning) and **EQ** (emotional intelligence)
- Uses Chinese Chain-of-Thought prompt template
- Evaluates on **test** split
- Metadata includes difficulty level information
""",  # noqa: E501
        dataset_id='AI-ModelScope/IQuiz',
        metric_list=['acc'],
        subset_list=['IQ', 'EQ'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT,
    )
)
class IQuizAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['choices'],
            target=record['answer'],
            metadata={'level': record.get('level', 'unknown')},
        )
