# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='aime24',
        pretty_name='AIME-2024',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'The AIME 2024 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model\'s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.',  # noqa: E501
        dataset_id='HuggingFaceH4/aime_2024',
        subset_list=['default'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=0,
        train_split=None,
        eval_split='train',  # Only train set is available
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class AIME24Adapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            metadata={
                'problem_id': record.get('id', ''),
                'solution': record.get('solution', ''),
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
