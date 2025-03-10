from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='simple_qa',
    pretty_name='SimpleQA',
    dataset_id='AI-ModelScope/SimpleQA',
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test')
class SimpleQAAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
