from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

@register_benchmark(
    BenchmarkMeta(
        name='olympiad_bench',
        pretty_name='OlympiadBench',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        "A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems.",
        dataset_id='knoveleng/OlympiadBench',
        subset_list=['default'],
        metric_list=['acc'],
        eval_split='train',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class OlympiadBenchAdapter(DefaultDataAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reformat_subset = True
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            target=record['answer'],
            metadata={
                'question_id': record['id'],
                'subfield': record['subfield'],
                'solution': record['solution'],
                'is_multiple_answer': record['is_multiple_answer'],
                'answer_type': record['answer_type'],
                'final_answer': record['final_answer'],
            },
        )