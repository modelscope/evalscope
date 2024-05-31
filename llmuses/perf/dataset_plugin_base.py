from abc import abstractmethod
import sys
from typing import Any, Dict, Iterator, List, Tuple
import json

class DatasetPluginBase:
    def __init__(self, 
                 dataset_path: str,
                 max_length: int = sys.maxsize, 
                 min_length: int = 0,):
        """Build data set plugin

        Args:
            dataset_path (str, optional): The input dataset path. Defaults to None.
        """
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.min_length = min_length

    def __next__(self):
        for item in self.build_prompt():
            yield item
        raise StopIteration

    def __iter__(self):
        return self.build_prompt()
    
    @abstractmethod
    def build_prompt(self)->Iterator[str]:
        """Build the request.

        Args:
            max_length (int, optional): The max prompt length by characters. Defaults to sys.maxsize.
            min_length (int, optional): The min prompt length by characters. Defaults to 0.

        Raises:
            NotImplementedError: The request is not impletion.

        Yields:
            Iterator[Dict]: Yield a request.
        """
        raise NotImplementedError
    
    def dataset_line_by_line(self, dataset: str)->Iterator[str]:
        """Get content line by line of dataset.

        Args:
            dataset (str): The dataset path.

        Yields:
            Iterator[str]: Each line of file.
        """
        with open(dataset, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
    
    def dataset_json_list(self, dataset: str)->Iterator[Dict]:
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