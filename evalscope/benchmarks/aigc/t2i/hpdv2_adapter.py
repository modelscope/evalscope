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
    name='hpdv2',
    dataset_id='AI-ModelScope/T2V-Eval-Prompts',
    model_adapter=OutputType.IMAGE_GENERATION,
    output_types=[OutputType.IMAGE_GENERATION],
    subset_list=['HPDv2'],
    metric_list=['HPSv2.1Score'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
)
class HPDv2Adapter(T2IBaseAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs) -> dict:
        if os.path.isfile(self.dataset_id):
            data_list = jsonl_to_list(self.dataset_id)
            data_dict = {self.subset_list[0]: {'test': data_list}}
            return data_dict
        else:
            return super().load(**kwargs)

    def get_gold_answer(self, input_d: dict) -> dict:
        # return prompt and elements dict
        return {'prompt': input_d.get('prompt'), 'tags': input_d.get('tags', {})}

    def match(self, gold: dict, pred: str) -> dict:
        # dummy match for general t2i
        # pred is the image path, gold is the prompt
        self._init_metrics()

        res = {}
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(images=[pred], texts=[gold['prompt']])[0][0]

            res[metric_name] = score.cpu().item()

            # fine-granular metrics
            category = gold['tags'].get('category')
            if category:
                res[f'{metric_name}_{category}'] = score.cpu().item()

        return res
