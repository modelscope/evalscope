# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='math_500',
        pretty_name='MATH-500',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        "MATH-500 is a benchmark for evaluating mathematical reasoning capabilities of AI models. It consists of 500 diverse math problems across five levels of difficulty, designed to test a model's ability to solve complex mathematical problems by generating step-by-step solutions and providing the correct final answer.",  # noqa: E501
        dataset_id='AI-ModelScope/MATH-500',
        subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class Math500Adapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            subset_key=f"Level {record['level']}",
            metadata={
                'question_id': record['unique_id'],
                'solution': record['solution'],
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
