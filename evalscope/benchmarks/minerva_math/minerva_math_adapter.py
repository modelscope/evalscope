from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='minerva_math',
        pretty_name='Minerva-Math',
        tags=[Tags.MATH, Tags.REASONING],
        description='Minerva-math is a benchmark designed to evaluate the mathematical and quantitative '
        'reasoning capabilities of LLMs. It consists of **272 problems** '
        'sourced primarily from **MIT OpenCourseWare** '
        'courses, covering advanced STEM subjects such as solid-state chemistry, astronomy, differential '
        'equations, and special relativity at the **university and graduate level**.',
        dataset_id='knoveleng/Minerva-Math',
        subset_list=['default'],
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        eval_split='train',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class MinervaMathAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._use_llm_judge = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['problem'],
            target=record['solution'],
            metadata={
                'type': record['type'],
                'idx': record['idx'],
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)
