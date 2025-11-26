# Copyright (c) Alibaba, Inc. and its affiliates.

import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

PROMPT_TEMPLATE = """{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.""".lstrip(
)  # noqa: E501

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}

""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='gsm8k',
        pretty_name='GSM8K',
        dataset_id='AI-ModelScope/gsm8k',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'GSM8K (Grade School Math 8K) is a dataset of grade school math problems, designed to evaluate the mathematical reasoning abilities of AI models.',  # noqa: E501
        subset_list=['main'],
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
class GSM8KAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        DELIM = '####'
        question = record['question']
        answer = record['answer'].split(DELIM)
        target = answer.pop().strip()
        reasoning = DELIM.join(answer)

        return Sample(input=question, target=target, metadata={'reasoning': reasoning.strip()})

    def sample_to_fewshot(self, sample: Sample) -> str:
        if sample.metadata:
            return (
                f'{sample.input}\n\nReasoning:\n' + f"{sample.metadata['reasoning']}\n\n"
                + f'ANSWER: \\boxed{{{sample.target}}}'
            )
        else:
            return ''

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
