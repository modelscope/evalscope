from abc import abstractmethod
import sys
from typing import Any, Dict, Iterator, List, Tuple
import json

class PerfPluginBase:
    @abstractmethod
    def build_request(self, 
                      model:str,
                      prompt: str=None, 
                      dataset: str=None,
                      max_length: int = sys.maxsize, 
                      min_length: int = 0,                       
                      **kwargs: Any)->Iterator[Dict]:
        """Build the request.

        Args:
            model (str): The request model.
            prompt (str): The input prompt, if not None, use prompt generate request. Defaults to None.
            dataset (str, optional): The input datasets. Defaults to None.
            max_length (int, optional): The max prompt length. Defaults to sys.maxsize.
            min_length (int, optional): The min prompt length. Defaults to 0.

        Raises:
            NotImplementedError: The request is not impletion.

        Yields:
            Iterator[Dict]: Yield a request.
        """
        raise NotImplementedError
    
    @abstractmethod
    def parse_responses(self, 
                        responses: List, 
                        request: Any=None,
                        **kwargs:Any) -> Tuple[int, int]:
        """Parser responses and return number of request and response tokens.

        Args:
            responses (List[bytes]): List of http response body, for stream output,
                there are multiple responses, each is bytes, for general only one. 

        Returns:
            Tuple: (Number of prompt_tokens and number of completion_tokens).
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