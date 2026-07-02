import json
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from evalscope.api.dataset.hub import download_dataset_file, load_dataset_from_hub
from evalscope.constants import HubType
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

    def supports_parallel_message_generation(self, total_count: Optional[int] = None) -> bool:
        """Return whether this dataset can build messages by independent index chunks."""
        return False

    def build_messages_parallel(self, total_count: int, workers: int) -> List[Any]:
        """Build messages using multiple worker processes.

        Dataset plugins should override this only when each output item can be
        generated independently and then reassembled by index without changing
        benchmark semantics.
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

    def load_hub_dataset(self, dataset_id: str, split: str = 'train', subset: str = 'default') -> Any:
        """Load a dataset from the configured data source.

        If dataset_path is a local directory, loads from there directly.
        Otherwise loads from ModelScope/HuggingFace based on data_source.

        Args:
            dataset_id (str): Remote dataset identifier (e.g. 'AI-ModelScope/LongAlpaca-12k').
            split (str): Dataset split to load (default: 'train').
            subset (str): Dataset subset/config name (default: 'default').

        Returns:
            A datasets.Dataset object.
        """
        dataset_path = self.query_parameters.dataset_path
        data_source = self.query_parameters.data_source or HubType.MODELSCOPE

        if dataset_path:
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"The specified dataset_path '{dataset_path}' does not exist.")
            data_id_or_path = dataset_path
            data_source = HubType.LOCAL
        else:
            data_id_or_path = dataset_id

        return load_dataset_from_hub(
            data_id_or_path=data_id_or_path,
            split=split,
            subset=subset,
            data_source=data_source,
        )

    def download_hub_file(self, dataset_id: str, file_name: str) -> str:
        """Download/resolve a single file from the configured data source.

        If dataset_path is an existing file, returns it directly.
        If dataset_path is a directory, looks for file_name inside it.
        Otherwise downloads from ModelScope/HuggingFace.

        Args:
            dataset_id (str): Remote dataset identifier (e.g. 'AI-ModelScope/HC3-Chinese').
            file_name (str): The file name to download or resolve.

        Returns:
            str: The resolved local file path.
        """
        dataset_path = self.query_parameters.dataset_path
        data_source = self.query_parameters.data_source or HubType.MODELSCOPE

        # dataset_path points to an existing file -> use directly
        if dataset_path and os.path.isfile(dataset_path):
            return dataset_path

        # dataset_path is a directory -> look for file inside
        if dataset_path and os.path.isdir(dataset_path):
            candidate = os.path.join(dataset_path, file_name)
            if os.path.isfile(candidate):
                return candidate
            # Fallback: treat directory as a hub-local dataset root
            return download_dataset_file(
                data_id_or_path=dataset_path,
                file_path=file_name,
                data_source=HubType.LOCAL,
            )

        # dataset_path is set but does not exist -> error
        if dataset_path:
            raise FileNotFoundError(f"The specified dataset_path '{dataset_path}' does not exist.")

        # Remote download
        return download_dataset_file(
            data_id_or_path=dataset_id,
            file_path=file_name,
            data_source=data_source,
        )
