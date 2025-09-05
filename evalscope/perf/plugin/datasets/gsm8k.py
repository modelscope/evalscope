import json
import os
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('gsm8k')
class Gsm8kDatasetPlugin(DatasetPluginBase):
    """
    Read dataset and return prompt.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        from modelscope.msdatasets import MsDataset
        dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='test')

        for item in dataset:
            prompt = item['question'].strip()
            if (
                len(prompt) > self.query_parameters.min_prompt_length
                and len(prompt) < self.query_parameters.max_prompt_length
            ):
                message = self.create_message(prompt)
                yield [message]
