import json
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.utils import load_tokenizer, tokenize_chat_messages

Message = Dict[str, Any]  # single OpenAI message: {"role": ..., "content": ...}
Messages = List[Message]  # delta messages for one turn


@dataclass
class Turn:
    """One turn within a multi-turn conversation.

    ``messages`` is the delta to append to the running context (typically one
    user / tool message).  Trace-replay datasets additionally set ``max_tokens``
    (per-turn output cap from the recorded length sequence) and
    ``tool_call_latency`` (seconds to sleep before sending this turn, simulating
    tool execution wait).  ``is_final`` flags the last turn of the trace.
    """

    messages: Messages
    max_tokens: Optional[int] = None
    tool_call_latency: Optional[float] = None
    is_final: bool = False


# Type alias for a full conversation (list of turn deltas).
Conversation = List[Turn]


class DatasetPluginBase:

    def __init__(self, query_parameters: Arguments):
        """Build data set plugin

        Args:
            dataset_path (str, optional): The input dataset path. Defaults to None.
        """
        self.query_parameters = query_parameters
        if query_parameters.tokenizer_path:
            self.tokenizer = load_tokenizer(query_parameters.tokenizer_path)
        else:
            self.tokenizer = None

    def __next__(self):
        for item in self.build_messages():
            yield item
        raise StopIteration

    def __iter__(self):
        return self.build_messages()

    @abstractmethod
    def build_messages(self) -> Iterator[Union[Messages, Conversation]]:
        """Build the request payload.

        Single-turn plugins yield a single-message list ``[{role, content}]``
        per request (``Messages``).  Multi-turn plugins yield a ``Conversation``
        (``List[Turn]``) per conversation; each ``Turn`` may carry per-turn
        ``max_tokens`` and ``tool_call_latency`` overrides.

        Raises:
            NotImplementedError: Subclass must implement.
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

    def create_message(self, text: str, image_urls: Optional[Union[List[str], str]] = None, role: str = 'user') -> Dict:
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

    def get_sampled_multi_turn_params(self) -> dict:
        """Return multi-turn parameters if ``multi_turn_args`` is set.

        Provides a common entry-point for all dataset plugins to obtain
        concrete multi-turn parameter values from
        :class:`~evalscope.perf.multi_turn_args.MultiTurnArgs`.

        Returns:
            Dict with field values, or an empty dict when ``multi_turn_args``
            is not configured.
        """
        if self.query_parameters.multi_turn_args:
            return self.query_parameters.multi_turn_args.sample_params()
        return {}

    def check_prompt_length(self, prompt: str) -> Tuple[bool, int]:
        """Check if the prompt length is within the specified range.

        When a tokenizer is available and apply_chat_template is enabled the prompt is
        wrapped in a chat message and the chat template is applied before counting tokens.
        This makes the client-side length measurement consistent with the token count
        that the server will report in usage.prompt_tokens (which includes the chat
        template overhead), and avoids filtering prompts that appear to be within range
        but actually exceed the target after the template is applied.

        Args:
            prompt (str): The input prompt string.

        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating whether the prompt is
                valid and its token/character length.
        """
        if self.tokenizer is None:
            prompt_length = len(prompt)
        elif self.query_parameters.apply_chat_template:
            messages = [self.create_message(prompt)]
            prompt_length = len(tokenize_chat_messages(self.tokenizer, messages))
        else:
            prompt_length = len(self.tokenizer.encode(prompt))
        is_valid = self.query_parameters.min_prompt_length <= prompt_length <= self.query_parameters.max_prompt_length
        return is_valid, prompt_length
