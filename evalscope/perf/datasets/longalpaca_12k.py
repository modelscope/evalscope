import sys
from typing import Any, Dict, Iterator, List
from evalscope.perf.dataset_plugin_base import DatasetPluginBase

from evalscope.perf.plugin_registry import register_dataset
from evalscope.perf.query_parameters import QueryParameters

@register_dataset('longalpaca')
class LongAlpacaDatasetPlugin(DatasetPluginBase):
    """Read data from file which is list of requests.
           Sample: https://huggingface.co/datasets/Yukang/LongAlpaca-12k
    """
    def __init__(self, query_parameters: QueryParameters):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        for item in self.dataset_json_list(self.query_parameters.dataset_path):
            prompt = item['instruction'].strip()
            if len(prompt) > self.query_parameters.min_prompt_length and len(prompt) < self.query_parameters.max_prompt_length:
                yield [{'role': 'user', 'content': prompt}]
