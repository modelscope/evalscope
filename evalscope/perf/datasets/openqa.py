from sys import maxsize
import sys
from typing import Any, Dict, Iterator, List
import json
from evalscope.perf.dataset_plugin_base import DatasetPluginBase
from evalscope.perf.plugin_registry import register_dataset
from evalscope.perf.query_parameters import QueryParameters

@register_dataset('openqa')
class OpenqaDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
        Datasets: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/blob/main/open_qa.jsonl
    """
    def __init__(self, query_parameters: QueryParameters):
        super().__init__(query_parameters)
        
    def build_messages(self) -> Iterator[List[Dict]]:
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            prompt = item['question'].strip()
            if len(prompt) > self.query_parameters.min_prompt_length and len(prompt) < self.query_parameters.max_prompt_length:
                yield [{'role': 'user', 'content': prompt}]
