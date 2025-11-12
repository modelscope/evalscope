# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='general_t2i',
        dataset_id='general_t2i',
        description='General Text-to-Image Benchmark',
        tags=[Tags.TEXT_TO_IMAGE, Tags.CUSTOM],
        subset_list=['default'],
        metric_list=['PickScore'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
    )
)
class GeneralT2IAdapter(Text2ImageAdapter):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def load_from_disk(self, **kwargs):
        if os.path.isfile(self.dataset_id):
            file_name = os.path.basename(self.dataset_id)
            file_without_ext = os.path.splitext(file_name)[0]
            self.subset_list = [file_without_ext]

        return super().load_from_disk(use_local_loader=True)

    def record_to_sample(self, record):
        return Sample(input=[ChatMessageUser(content=record['prompt'])], metadata={'image_path': record['image_path']})
