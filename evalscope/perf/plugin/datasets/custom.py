import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase, Message, Messages
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils import get_logger

logger = get_logger()

# Internal key used to carry tools definitions through the conversation pipeline.
# The dataset plugin embeds tools in the first message; the API plugin extracts
# and injects them into the request body, then strips the key before sending.
_TOOL_CONTEXT_KEY = "__evalscope_tools__"


@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            prompt = item.strip()
            is_valid, _ = self.check_prompt_length(prompt)
            if is_valid:
                if self.query_parameters.apply_chat_template:
                    message = self.create_message(prompt)
                    yield [message]
                else:
                    yield prompt


@register_dataset('custom_multi_turn')
class CustomMultiTurnDatasetPlugin(DatasetPluginBase):
    """Multi-turn dataset plugin that reads conversations from a local JSONL file.

    Each line in the JSONL file must be a complete conversation represented as a
    JSON array of OpenAI-style message dicts, for example::

        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "How are you?"}]

    ``build_messages`` yields each conversation as a ``List[Messages]`` where
    every ``Messages`` is the delta for one turn (all non-assistant messages
    between two consecutive assistant messages).  The runner appends real
    model responses; dataset assistant content is discarded.

    ``--max-turns`` optionally truncates long conversations to at most N turns.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        if not query_parameters.dataset_path:
            raise ValueError(
                'custom_multi_turn requires --dataset-path to point to a local JSONL file. '
                'Each line must be a JSON array of OpenAI message dicts.'
            )

    def _split_into_turns(self, messages: List[Message]) -> List[Messages]:
        """Split a flat message list into per-turn delta lists.

        Turn boundaries are ``assistant`` messages that mark the end of a
        conversational turn (i.e., followed by a ``user`` message or at the
        end of the conversation).  ``assistant`` messages that are part of a
        tool-calling chain (followed by ``tool`` responses) are retained in the
        current turn so that ``tool`` messages have their corresponding
        ``assistant`` ``tool_calls`` in the context.

        Example (standard conversation)::
            [system, user_1, assistant_ref, user_2, assistant_ref, user_3]
            -> [[system, user_1], [user_2], [user_3]]

        Example (agent with tool calls)::
            [system, user, assistant(tool_calls), tool, assistant(content)]
            -> [[system, user], [assistant(tool_calls), tool, assistant(content)]]

        Args:
            messages: Flat list of OpenAI message dicts.

        Returns:
            ``List[Messages]`` – one ``Messages`` per turn.
        """
        turns: List[Messages] = []
        current: Messages = []
        for idx, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                # Check if this assistant is followed by a tool response.
                # If so, it's part of a tool-calling chain and should be kept.
                next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
                if next_msg and next_msg.get('role') == 'tool':
                    # Part of tool-calling chain – keep in current turn.
                    current.append(msg)
                else:
                    # Turn boundary – finalize current turn and discard this
                    # assistant (the runner will append the model's real output).
                    if current:
                        turns.append(current)
                        current = []
            else:
                current.append(msg)
        if current:
            turns.append(current)
        return turns


def _embed_tools(turn_delta: Messages, tools: List[Dict]) -> None:
    """Embed tools definitions into the first message of a turn delta.

    Uses an internal key that the API plugin extracts and injects into the
    request body, then strips before sending.
    """
    if turn_delta:
        first_msg = turn_delta[0]
        first_msg[_TOOL_CONTEXT_KEY] = tools

    def build_messages(self) -> Iterator[List[Messages]]:
        """Yield complete conversations from a JSONL or JSON file.

        Supported file formats:
        - **JSONL**: one JSON array per line (existing behavior).
        - **JSON**: a top-level JSON array of conversations, one per element.

        Each yielded item is a ``List[Messages]`` where every ``Messages``
        contains the delta for one turn.  The multi-turn benchmark runner
        extends the growing context with each delta and appends the model's
        real response after each turn.

        If tools definitions are present in the source data, they are embedded
        into the first message of the first turn using an internal key that is
        later extracted by the API plugin and injected into the request body.
        """
        max_turns = self.query_parameters.max_turns

        for messages, tools in self._iter_conversations():
            turns = self._split_into_turns(messages)

            # Embed tools into the first turn so they flow through the pipeline.
            if tools and turns:
                _embed_tools(turns[0], tools)

            # Apply max_turns truncation at the dataset layer
            if max_turns is not None:
                turns = turns[:max_turns]

            # A valid multi-turn conversation needs at least one turn
            if not turns:
                continue

            # Length filter: check the first user message of the first turn as a proxy
            first_msg = next((m for m in turns[0] if m.get('role') == 'user'), None)
            if first_msg is None:
                yield turns
                continue

            first_user_content = first_msg['content']
            if not isinstance(first_user_content, str):
                # Vision messages or other non-text content: skip length check
                yield turns
                continue

            is_valid, _ = self.check_prompt_length(first_user_content)
            if is_valid:
                yield turns

    def _iter_conversations(self) -> Iterator[Tuple[List[Message], Optional[List[Dict]]]]:
        """Iterate (messages, tools) tuples from the dataset file, auto-detecting format.

        Supported formats (auto-detected):
        1. **Single JSON object**: ``{"model": "...", "messages": [...], "tools": [...]}``
           – extracts ``messages`` and ``tools``.
        2. **JSON array of conversation objects**: ``[{"messages": [...], "tools": [...]}, ...]``.
        3. **JSON array of message arrays**: ``[[...], [...]]`` – no tools.
        4. **JSONL**: one JSON array (or conversation object) per line.
        """
        path = self.query_parameters.dataset_path

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not valid JSON – fall back to JSONL
            yield from self._iter_jsonl(path)
            return

        if isinstance(data, dict):
            # Single conversation object
            messages = data.get('messages')
            if messages and isinstance(messages, list):
                tools = data.get('tools')
                if tools and isinstance(tools, list):
                    yield messages, tools
                else:
                    yield messages, None
            return

        if isinstance(data, list):
            if not data:
                return
            first = data[0]

            if isinstance(first, dict):
                # Array of conversation objects: each may have "messages" and "tools"
                for item in data:
                    if isinstance(item, dict):
                        messages = item.get('messages')
                        if messages and isinstance(messages, list):
                            tools = item.get('tools')
                            if tools and isinstance(tools, list):
                                yield messages, tools
                            else:
                                yield messages, None
                    elif isinstance(item, list) and item:
                        yield item, None
                return

            if isinstance(first, list):
                # Array of message arrays
                for conv in data:
                    if isinstance(conv, list) and conv:
                        yield conv, None
                return

        # Fallback
        logger.warning(f'Unsupported JSON structure in {path}, falling back to JSONL')
        yield from self._iter_jsonl(path)

    def _iter_jsonl(self) -> Iterator[Tuple[List[Message], Optional[List[Dict]]]]:
        """Read JSONL file: one JSON array (or conversation object) per line."""
        path = self.query_parameters.dataset_path
        for line in self.dataset_line_by_line(path):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f'Skipping malformed JSON line: {e}')
                continue

            # Support both raw message array and conversation object in JSONL
            if isinstance(item, dict):
                messages = item.get('messages')
                if not messages or not isinstance(messages, list):
                    continue
                tools = item.get('tools')
                if not tools or not isinstance(tools, list):
                    tools = None
            elif isinstance(item, list):
                messages = item
                tools = None
            else:
                continue

            if not messages:
                logger.warning('Skipping line: empty messages array.')
                continue

            # Validate that every element has role and content fields
            if not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                logger.warning('Skipping line: each message must have "role" and "content" fields.')
                continue

            yield messages, tools


if __name__ == '__main__':
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    args = Arguments(
        model='qwen2.5-7b-instruct',
        url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
        dataset_path='outputs/perf_data.txt',
        api_key='EMPTY',
        dataset='custom',
    )

    run_perf_benchmark(args)
