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
    name='general_t2i',
    dataset_id='general_t2i',
    model_adapter=OutputType.IMAGE_GENERATION,
    output_types=[OutputType.IMAGE_GENERATION],
    subset_list=['default'],
    metric_list=['PickScore'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
)
class GeneralT2IAdapter(T2IBaseAdapter):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def load(self, dataset_name_or_path: str = None, subset_list: list = None, **kwargs) -> dict:
        dataset_name_or_path = dataset_name_or_path or self.dataset_id
        subset_list = subset_list or self.subset_list

        data_file_dict = defaultdict(str)
        data_item_dict = defaultdict(list)

        # get data file path and subset name
        if os.path.isdir(dataset_name_or_path):
            for subset_name in subset_list:
                data_file_dict[subset_name] = os.path.join(dataset_name_or_path, f'{subset_name}.jsonl')
        elif os.path.isfile(dataset_name_or_path):
            cur_subset_name = os.path.splitext(os.path.basename(dataset_name_or_path))[0]
            data_file_dict[cur_subset_name] = dataset_name_or_path
        else:
            raise ValueError(f'Invalid dataset path: {dataset_name_or_path}')

        # load data from local disk
        try:
            for subset_name, file_path in data_file_dict.items():
                data_item_dict[subset_name] = jsonl_to_list(file_path)
        except Exception as e:
            raise ValueError(f'Failed to load data from {self.dataset_id}, got error: {e}')

        data_dict = {subset_name: {'test': data_item_dict[subset_name]} for subset_name in data_file_dict.keys()}

        return data_dict
