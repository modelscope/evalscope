import json
import subprocess
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('openqa')
class OpenqaDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/resolve/master/open_qa.jsonl
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        if not self.query_parameters.dataset_path:
            subprocess.call([
                'modelscope',
                'download',
                '--dataset',
                'AI-ModelScope/HC3-Chinese',
                'open_qa.jsonl',
                '--local_dir',
                './data',
            ])
            self.query_parameters.dataset_path = './data/open_qa.jsonl'

        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            prompt = item['question'].strip()
            if (len(prompt) > self.query_parameters.min_prompt_length
                    and len(prompt) < self.query_parameters.max_prompt_length):
                yield [{'role': 'user', 'content': prompt}]
