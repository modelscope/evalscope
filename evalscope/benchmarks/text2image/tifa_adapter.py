# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='tifa160',
        pretty_name='TIFA-160',
        dataset_id='AI-ModelScope/T2V-Eval-Prompts',
        description='TIFA-160 Text-to-Image Benchmark',
        tags=[Tags.TEXT_TO_IMAGE],
        subset_list=['TIFA-160'],
        metric_list=['PickScore'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class TIFA_Adapter(Text2ImageAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
