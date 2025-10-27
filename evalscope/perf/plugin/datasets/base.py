import json
import sys
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Tuple, Union

from evalscope.perf.arguments import Arguments


class DatasetPluginBase:

    def __init__(self, query_parameters: Arguments):
        """Build data set plugin

        Args:
            dataset_path (str, optional): The input dataset path. Defaults to None.
        """
        self.query_parameters = query_parameters
        if query_parameters.tokenizer_path:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(query_parameters.tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = None

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

    def create_message(self, text: str, image_urls: Union[List[str], str] = None, role: str = 'user') -> Dict:
        """Create a message with text and optional image URLs.

        Args:
            text (str): The text content of the message.
            image_urls (List[str], optional): List of image URLs. Defaults to None.
            role (str, optional): The role of the message sender. Defaults to "user".

        Returns:
            Dict: A dictionary representing the message.
        """
        if image_urls is None:
            message = {'role': role, 'content': text}
        else:
            message = {'role': role, 'content': [{'type': 'text', 'text': text}]}
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            for url in image_urls:
                message['content'].append({'type': 'image_url', 'image_url': {'url': url}})
        return message

    def check_prompt_length(self, prompt: str) -> Tuple[bool, int]:
        """Check if the prompt length is within the specified range.

        Args:
            prompt (str): The input prompt string.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating whether the prompt is valid and its length.
        """
        if self.tokenizer is None:
            prompt_length = len(prompt)
        else:
            prompt_length = len(self.tokenizer.encode(prompt))
        is_valid = self.query_parameters.min_prompt_length <= prompt_length <= self.query_parameters.max_prompt_length
        return is_valid, prompt_length
