from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='amc',
        pretty_name='AMC',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        'AMC (American Mathematics Competitions) is a series of mathematics competitions for high school students.',
        dataset_id='evalscope/amc_22-24',
        subset_list=['amc22', 'amc23', 'amc24'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class AMCAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use split as subset
        self.split_as_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['answer'],
            metadata={
                'year': record['year'],
                'url': record['url'],
                'solution': record.get('solution', '')
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
