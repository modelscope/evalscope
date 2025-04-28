# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from collections import defaultdict
from typing import List, Optional, Union

from evalscope.benchmarks import Benchmark
from evalscope.constants import OutputType
from evalscope.metrics import mean
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger
from .base import T2IBaseAdapter

logger = get_logger()


@Benchmark.register(
    name='evalmuse',
    dataset_id='AI-ModelScope/T2V-Eval-Prompts',
    model_adapter=OutputType.IMAGE_GENERATION,
    output_types=[OutputType.IMAGE_GENERATION],
    subset_list=['EvalMuse'],
    metric_list=['FGA_BLIP2Score'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
)
class EvalMuseAdapter(T2IBaseAdapter):

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
        res = {}
        for metric_name, metric_func in self.metrics.items():
            if metric_name == 'FGA_BLIP2Score':
                # For FGA_BLIP2Score, we need to pass the dictionary
                score = metric_func(images=[pred], texts=[gold])[0][0]
            else:
                score = metric_func(images=[pred], texts=[gold['prompt']])[0][0]
            if isinstance(score, dict):
                for k, v in score.items():
                    res[f'{metric_name}:{k}'] = v.cpu().item()
            else:
                res[metric_name] = score.cpu().item()
        return res

    def compute_metric(self, review_res_list: Union[List[dict], List[List[dict]]], **kwargs) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples
        """
        items = super().compute_dict_metric(review_res_list, **kwargs)
        # add statistics for each metric
        new_items = defaultdict(list)
        for metric_name, value_list in items.items():
            if 'FGA_BLIP2Score' in metric_name and '(' in metric_name:  # FGA_BLIP2Score element score
                metrics_prefix = metric_name.split(':')[0]
                category = metric_name.rpartition('(')[-1].split(')')[0]
                new_items[f'{metrics_prefix}:{category}'].extend(value_list)
            else:
                new_items[metric_name].extend(value_list)

        # calculate mean for each metric
        return [{'metric_name': k, 'score': mean(v), 'num': len(v)} for k, v in new_items.items()]
