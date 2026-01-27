# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """
{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.

""".lstrip()  # noqa: E501

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}

""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='mgsm',
        pretty_name='MGSM',
        dataset_id='evalscope/mgsm',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTI_LINGUAL],
        description="""
## Overview

MGSM (Multilingual Grade School Math) is a benchmark designed to evaluate multilingual mathematical reasoning capabilities of language models. It extends GSM8K to 11 typologically diverse languages, testing whether models can perform chain-of-thought reasoning across different languages.

## Task Description

- **Task Type**: Multilingual Mathematical Word Problem Solving
- **Input**: Grade school math word problem in one of 11 languages
- **Output**: Step-by-step reasoning with numerical answer
- **Languages**: English, Spanish, French, German, Russian, Chinese, Japanese, Thai, Swahili, Bengali, Telugu

## Key Features

- 250 problems per language (translated from GSM8K)
- 11 typologically diverse languages covering different language families
- Tests multilingual chain-of-thought reasoning capabilities
- Same problem content across languages for cross-lingual comparison
- Designed to evaluate language-agnostic mathematical reasoning

## Evaluation Notes

- Default configuration uses **4-shot** examples
- Answers should be formatted within `\\boxed{}` for proper extraction
- Use `subset_list` to evaluate specific languages (e.g., `['en', 'zh', 'ja']`)
- Cross-lingual performance comparison supported
- Few-shot examples are drawn from the train split in the same language
""",
        subset_list=['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te'],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class MGSMAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        question = record['question']
        target = str(record['answer_number'])

        return Sample(
            input=question,
            target=target,
            metadata={
                'reasoning': record['answer'],
                'equation_solution': record['equation_solution'],
            }
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        if sample.metadata:
            return (
                f'{sample.input}\n\nReasoning:\n' + f"{sample.metadata['reasoning']}\n\n" + f'ANSWER: {sample.target}'
            )
        else:
            return ''

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
