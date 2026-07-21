import os
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.dataset_args import TextDatasetArgs
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('longalpaca')
class LongAlpacaDatasetPlugin(DatasetPluginBase):
    """Read data from file which is list of requests.
           Sample: https://www.modelscope.cn/datasets/AI-ModelScope/LongAlpaca-12k/files
    """

    args_schema = TextDatasetArgs

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        if self.query_parameters.dataset_path and os.path.isfile(self.query_parameters.dataset_path):
            ds = self.dataset_json_list(self.query_parameters.dataset_path)
        else:
            ds = self.load_hub_dataset(dataset_id='AI-ModelScope/LongAlpaca-12k', split='train')
        for item in ds:
            prompt = item['instruction'].strip()
            prompt = self.prepare_prompt(prompt)
            if prompt is None:
                continue
            if self.query_parameters.apply_chat_template:
                yield [self.create_message(prompt)]
            else:
                yield prompt
