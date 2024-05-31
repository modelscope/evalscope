import sys
from typing import Dict, Iterator
from llmuses.perf.dataset_plugin_base import DatasetPluginBase
from llmuses.perf.plugin_registry import register_dataset

@register_dataset('line_by_line')
class LineByLineDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    """
    def __init__(self, 
                 dataset_path: str,
                 max_length: int = sys.maxsize, 
                 min_length: int = 0,):
        super().__init__(dataset_path, max_length, min_length)

    def build_prompt(self) -> Iterator[Dict]:
        for item in self.dataset_line_by_line(self.dataset_path):
            prompt = item.strip()
            if len(prompt) > self.min_length and len(prompt) < self.max_length:
                yield prompt
