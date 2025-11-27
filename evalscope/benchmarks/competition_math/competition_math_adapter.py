# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) EleutherAI, Inc. and its affiliates.

from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

PROMPT_TEMPLATE = """
Problem:
{question}

Please reason step by step, and put your final answer within \\boxed{{}}.
""".lstrip()

FEWSHOT_TEMPLATE = """
Here are some examples of how to solve similar problems:

{fewshot}
""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='competition_math',
        pretty_name='Competition-MATH',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'The MATH (Mathematics) benchmark is designed to evaluate the mathematical reasoning abilities of AI models through a variety of problem types, including arithmetic, algebra, geometry, and more.',
        dataset_id='evalscope/competition_math',
        subset_list=['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class CompetitionMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        from evalscope.metrics.math_parser import extract_answer

        return Sample(
            input=record['problem'],
            target=extract_answer(record['solution']),
            subset_key=record['level'],
            metadata={
                'reasoning': record.get('solution', ''),
                'type': record.get('type', ''),
            },
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        return f'Problem:\n{sample.input}\nSolution:\n{sample.target}'

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
