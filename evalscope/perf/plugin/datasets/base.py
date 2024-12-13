import json
import sys
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Tuple

from evalscope.perf.arguments import Arguments


class DatasetPluginBase:

    def __init__(self, query_parameters: Arguments):
        """Build data set plugin

        Args:
            dataset_path (str, optional): The input dataset path. Defaults to None.
        """
        self.query_parameters = query_parameters

    def __next__(self):
        for item in self.build_messages():
            yield item
        raise StopIteration

    def __iter__(self):
        return self.build_messages()

    @abstractmethod
    def build_messages(self) -> Iterator[List[Dict]]:
        """Build the request.

        Raises:
            NotImplementedError: The request is not impletion.

        Yields:
            Iterator[List[Dict]]: Yield request messages.
        """
        raise NotImplementedError

    def dataset_line_by_line(self, dataset: str) -> Iterator[str]:
        """Get content line by line of dataset.

        Args:
            dataset (str): The dataset path.

        Yields:
            Iterator[str]: Each line of file.
        """
        with open(dataset, 'r', encoding='utf-8') as f:
            for line in f:
                yield line

    def dataset_json_list(self, dataset: str) -> Iterator[Dict]:
        """Read data from file which is list of requests.
           Sample: https://huggingface.co/datasets/Yukang/LongAlpaca-12k

        Args:
            dataset (str): The dataset path.

        Yields:
            Iterator[Dict]: The each request object.
        """
        with open(dataset, 'r', encoding='utf-8') as f:
            content = f.read()
        data = json.loads(content)
        for item in data:
            yield item
