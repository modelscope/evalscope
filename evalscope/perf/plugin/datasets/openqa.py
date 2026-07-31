import json
import os
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.dataset_args import TextDatasetArgs
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('openqa')
class OpenqaDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://www.modelscope.cn/datasets/AI-ModelScope/HC3-Chinese/resolve/master/open_qa.jsonl
    """

    args_schema = TextDatasetArgs

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        if not self.query_parameters.dataset_path or os.path.isdir(self.query_parameters.dataset_path):
            self.query_parameters.dataset_path = self.download_hub_file(
                dataset_id='AI-ModelScope/HC3-Chinese', file_name='open_qa.jsonl'
            )

        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            prompt = item['question'].strip()
            result = self.prepare_messages(prompt)
            if result is None:
                continue
            yield result
