from typing import Any, Dict, Iterator, List

from modelscope import MsDataset

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('longalpaca')
class LongAlpacaDatasetPlugin(DatasetPluginBase):
    """Read data from file which is list of requests.
           Sample: https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/files
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        if not self.query_parameters.dataset_path:
            ds = MsDataset.load('AI-ModelScope/LongAlpaca-12k', subset_name='default', split='train')
        else:
            ds = self.dataset_json_list(self.query_parameters.dataset_path)
        for item in ds:
            prompt = item['instruction'].strip()
            if len(prompt) > self.query_parameters.min_prompt_length and len(
                    prompt) < self.query_parameters.max_prompt_length:
                yield [{'role': 'user', 'content': prompt}]
