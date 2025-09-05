import json
import os
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('cmmlu')
class CmmluDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    Datasets: https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu-test.jsonl
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        if not self.query_parameters.dataset_path:
            from modelscope import dataset_snapshot_download

            file_name = 'cmmlu-test.jsonl'
            local_path = dataset_snapshot_download('haonan-li/cmmlu', allow_patterns=[file_name])
            self.query_parameters.dataset_path = os.path.join(local_path, file_name)

        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            prompt = item['question'].strip()

            # 根据长度过滤
            if not (self.query_parameters.min_prompt_length < len(prompt) < self.query_parameters.max_prompt_length):
                continue

            if self.query_parameters.apply_chat_template:
                message = self.create_message(prompt)
                yield [message]
            else:
                yield [prompt]
