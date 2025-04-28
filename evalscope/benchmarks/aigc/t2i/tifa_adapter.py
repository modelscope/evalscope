# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from collections import defaultdict
from typing import List, Optional, Union

from evalscope.benchmarks import Benchmark
from evalscope.constants import OutputType
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger
from .base import T2IBaseAdapter

logger = get_logger()


@Benchmark.register(
    name='tifa160',
    dataset_id='AI-ModelScope/T2V-Eval-Prompts',
    model_adapter=OutputType.IMAGE_GENERATION,
    output_types=[OutputType.IMAGE_GENERATION],
    subset_list=['TIFA-160'],
    metric_list=['PickScore'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
)
class TIFA_Adapter(T2IBaseAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs) -> dict:
        if os.path.isfile(self.dataset_id):
            data_list = jsonl_to_list(self.dataset_id)
            data_dict = {self.subset_list[0]: {'test': data_list}}
            return data_dict
        else:
            return super().load(**kwargs)
