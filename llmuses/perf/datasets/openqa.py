from sys import maxsize
import sys
from typing import Any, Dict, Iterator
import json
from llmuses.perf.dataset_plugin_base import DatasetPluginBase
from llmuses.perf.plugin_registry import register_dataset

@register_dataset('openqa')
class OpenqaDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
        Datasets: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/blob/main/open_qa.jsonl
    """
    def __init__(self, 
                 dataset_path: str,
                 max_length: int = sys.maxsize, 
                 min_length: int = 0,):
        super().__init__(dataset_path, max_length, min_length)
        
    def build_prompt(self) -> Iterator[Dict]:
        for item in self.dataset_line_by_line(self.dataset_path):
            item = json.loads(item)
            prompt = item['question'].strip()
            if len(prompt) > self.min_length and len(prompt) < self.max_length:
                yield prompt
