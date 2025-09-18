from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

@register_benchmark(
    BenchmarkMeta(
        name='amc_23',
        pretty_name='AMC_23',
        tags=[Tags.MATH, Tags.REASONING],
        description=
        "The AMC23 dataset is a public dataset containing mathematical problems and their solutions, primarily intended for research and development of AI models related to mathematics.",
        dataset_id='knoveleng/AMC-23',
        subset_list=['default'],
        metric_list=['acc'],
        eval_split='train',
        prompt_template='{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
    )
)
class AMC_23Adapter(DefaultDataAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reformat_subset = True
    
    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            target=record['answer'],
            metadata={
                'question_id': record['id'],
                'url': record['url'],
                'problem': record['problem'],
            },
        )