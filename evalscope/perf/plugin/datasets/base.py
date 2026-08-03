import json
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Type, Union

from evalscope.api.dataset.hub import download_dataset_file, load_dataset_from_hub
from evalscope.constants import HubType
from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.dataset_args import BaseDatasetArgs, TextLengthArgs
from evalscope.perf.plugin.datasets.utils import (
    fit_prefix_to_budget,
    fit_text_to_token_len,
    load_tokenizer,
    tokenize_chat_messages,
)
from evalscope.utils.logger import get_logger

logger = get_logger()

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

    args_schema: ClassVar[Type[BaseDatasetArgs]] = BaseDatasetArgs
    """Pydantic schema for this dataset's ``--dataset-args``.

    Subclasses override this to declare their own typed argument model.  The
    default empty schema rejects any ``--dataset-args`` keys (fail-fast).
    """

    provides_arrival_schedule: bool = False
    """True if the dataset embeds per-request arrival times, bypassing ``--rate``."""

    requires_model: bool = True
    """False if the dataset carries model info in each request body, making ``--model`` optional."""

    def __init__(self, query_parameters: Arguments):
        """Build data set plugin

        Args:
            dataset_path (str, optional): The input dataset path. Defaults to None.
        """
        self.query_parameters = query_parameters
        # Validate the raw --dataset-args against this plugin's own schema.  The
        # plugin owns the schema, so no registry lookup (and no import cycle with
        # the config layer) is needed.
        self.dataset_args = self.args_schema(**(query_parameters.dataset_args or {}))
        if query_parameters.tokenizer_path:
            self.tokenizer = load_tokenizer(query_parameters.tokenizer_path)
        else:
            self.tokenizer = None
        if (
            isinstance(self.dataset_args, TextLengthArgs) and self.dataset_args.target_input_len is not None
            and self.tokenizer is None
        ):
            raise ValueError('`target_input_len` requires a tokenizer; please set --tokenizer-path.')
        self._prefix_ids: Optional[List[int]] = None
        if isinstance(self.dataset_args, TextLengthArgs) and self.dataset_args.prefix_file is not None:
            self._prefix_ids = self._load_prefix_ids(self.dataset_args.prefix_file)
        self._warned_conversation_dropped: bool = False

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
        """Return multi-turn parameters if configured.

        Reads from the resolved ``dataset_args`` when it is a
        :class:`~evalscope.perf.multi_turn_args.MultiTurnArgs` (the new
        ``--dataset-args`` path), otherwise falls back to the deprecated
        ``--multi-turn-args`` field.

        Returns:
            Dict with field values, or an empty dict when neither is set.
        """
        mt_args = self._get_multi_turn_args()
        return mt_args.sample_params() if mt_args else {}

    def _get_multi_turn_args(self):
        """Resolve the effective MultiTurnArgs (new dataset_args or legacy field)."""
        from evalscope.perf.multi_turn_args import MultiTurnArgs
        if isinstance(self.dataset_args, MultiTurnArgs):
            return self.dataset_args
        return self.query_parameters.multi_turn_args

    def prepare_prompt(self, prompt: str) -> Optional[str]:
        """Apply the input-length policy to a single-turn text prompt.

        When ``target_input_len`` is not set (via ``--dataset-args``), falls back
        to the existing ``min/max_prompt_length`` filter and returns the prompt
        unchanged when valid (``None`` when out of range).  When it is set, the
        prompt is fit to the target length per ``input_len_mode`` (``cap`` /
        ``drop``); ``None`` means the caller should skip this item.

        Args:
            prompt (str): The raw prompt text.

        Returns:
            Optional[str]: The adjusted prompt, or ``None`` to skip.
        """
        args = self.dataset_args
        if not isinstance(args, TextLengthArgs) or args.target_input_len is None:
            is_valid, _ = self.check_prompt_length(prompt)
            return prompt if is_valid else None
        if self.tokenizer is None:
            raise ValueError('`target_input_len` requires a tokenizer; please set --tokenizer-path.')
        return fit_text_to_token_len(prompt, args.target_input_len, args.input_len_mode, self.tokenizer)

    def _load_prefix_ids(self, prefix_file: str) -> List[int]:
        """Read and tokenize the long-context prefix file once at construction time.

        Emits one-shot warnings for the tiling and no-chat-template downgrade
        cases so per-request generation stays silent.
        """
        args = self.dataset_args
        if not os.path.isfile(prefix_file):
            raise FileNotFoundError(f"The specified prefix_file '{prefix_file}' does not exist.")
        with open(prefix_file, 'r', encoding='utf-8') as f:
            prefix_text = f.read()
        ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"The specified prefix_file '{prefix_file}' is empty.")
        if len(ids) < args.target_input_len:
            logger.warning(
                f'prefix_file has {len(ids)} tokens, fewer than target_input_len={args.target_input_len}; '
                'the prefix will be repeated (tiled) to fill the remaining budget when needed.'
            )
        if args.prefix_role == 'system' and not self.query_parameters.apply_chat_template:
            logger.warning(
                "prefix_role='system' requires a chat template; falling back to plain-text "
                'prefix concatenation because apply_chat_template is disabled.'
            )
        return ids

    def content_token_len(self, content: str) -> int:
        """Return the bare content token count of one message (no special tokens)."""
        return len(self.tokenizer.encode(content, add_special_tokens=False))

    def measure_messages_len(self, messages: Messages) -> int:
        """Return the total bare content token count of a messages list.

        Every message counts, so multi-turn history is included in the length
        measurement instead of only the last user turn.
        """
        return sum(self.content_token_len(message['content']) for message in messages)

    def apply_prefix_to_messages(self, messages: Messages) -> Messages:
        """Inject the configured long-context prefix into an OpenAI messages list.

        The prefix budget is ``target_input_len`` minus the bare content tokens
        of **all** messages, so prefix plus conversation together occupy the
        target length.  ``prefix_role='system'`` prepends a system message
        (chat-template mode only), whose content is separated from the prompt by
        template markers; otherwise the prefix is prepended to the first message
        content so it stays at the very front of the token stream (prefix-cache
        friendly).  The prefix and prompt are counted independently, so on BPE
        tokenizers the total may differ from the target by ~1 token when
        characters merge across the join (chat-template mode is exact).  No-op
        when ``prefix_file`` is not configured or the remaining budget is zero.
        """
        if self._prefix_ids is None:
            return messages
        budget = self.dataset_args.target_input_len - self.measure_messages_len(messages)
        if budget <= 0:
            return messages
        prefix = fit_prefix_to_budget(self._prefix_ids, budget, self.tokenizer)
        if not prefix:
            return messages
        if self.dataset_args.prefix_role == 'system' and self.query_parameters.apply_chat_template:
            return [self.create_message(prefix, role='system')] + messages
        messages[0]['content'] = prefix + messages[0]['content']
        return messages

    def prepare_messages(self, prompt: str) -> Optional[Union[str, Messages]]:
        """Apply length policy, prefix injection and chat wrapping to one prompt.

        Combines :meth:`prepare_prompt` (cap/drop fitting) with the long-context
        prefix injection (issue #1524).  Returns ``None`` when the prompt should
        be skipped, a plain string when ``apply_chat_template`` is off,
        otherwise an OpenAI messages list.
        """
        prepared = self.prepare_prompt(prompt)
        if prepared is None:
            return None
        if not self.query_parameters.apply_chat_template:
            if self._prefix_ids is None:
                return prepared
            # Plain-text endpoint: fill the remaining budget with the prefix. The
            # prefix and prompt are counted independently, so the total may drift
            # by ~1 token when characters merge across the join.
            budget = self.dataset_args.target_input_len - self.content_token_len(prepared)
            return fit_prefix_to_budget(self._prefix_ids, budget, self.tokenizer) + prepared
        return self.apply_prefix_to_messages([self.create_message(prepared)])

    def prepare_conversation(self, messages: Messages) -> Optional[Messages]:
        """Apply the input-length policy and prefix injection to a conversation.

        Unlike :meth:`prepare_messages` the length is measured over every message
        content, so the history counts towards ``target_input_len``.  A
        conversation already exceeding the target is dropped rather than
        truncated: cutting the history or the last user turn would either
        destroy the dialogue or silently change what is being benchmarked.
        Shorter conversations are filled up by the configured prefix.

        ``input_len_mode='drop'`` is intentionally not honoured here: a
        conversation whose summed content lands on the target exactly is so rare
        that it would drop the whole dataset, so callers reject that combination
        up front (see ``ShareGPTDatasetPluginBase``).

        Args:
            messages (Messages): The conversation, ending with a user message.

        Returns:
            Optional[Messages]: The adjusted conversation, or ``None`` to skip it.
        """
        args = self.dataset_args
        if not isinstance(args, TextLengthArgs) or args.target_input_len is None:
            # Legacy path: filter on the last user turn only.
            is_valid, _ = self.check_prompt_length(messages[-1]['content'])
            return messages if is_valid else None
        if self.tokenizer is None:
            raise ValueError('`target_input_len` requires a tokenizer; please set --tokenizer-path.')

        total_len = self.measure_messages_len(messages)
        if total_len > args.target_input_len:
            if not self._warned_conversation_dropped:
                self._warned_conversation_dropped = True
                logger.warning(
                    f'Dropping conversations whose total content length exceeds target_input_len='
                    f'{args.target_input_len} (first one had {total_len} tokens); the length is measured '
                    'over all turns, so long histories are skipped instead of truncated.'
                )
            return None
        return self.apply_prefix_to_messages(messages)

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
