import re
from lark import v_args

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.utils.logger import get_logger

logger = get_logger()


@Benchmark.register(
    name='live_code_bench',
    pretty_name='Live Code Bench',
    dataset_id='modelscope/humaneval',
    subset_list=['v1'],
    metric_list=['Pass@1'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    prompt_template='Complete the following python code:\n{query}',
)
class LiveCodeBenchAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
