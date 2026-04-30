import json
from typing import Any, Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase, Message, Messages
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils import get_logger

logger = get_logger()


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

        Uses ``assistant`` messages as turn boundaries.  Each run of
        non-assistant messages before an ``assistant`` message (or at the end
        of the conversation) forms one turn's delta.

        Example::
            [system, user_1, assistant_ref, user_2, assistant_ref, user_3]
            -> [[system, user_1], [user_2], [user_3]]

        Args:
            messages: Flat list of OpenAI message dicts.

        Returns:
            ``List[Messages]`` – one ``Messages`` per turn.
        """
        turns: List[Messages] = []
        current: Messages = []
        for msg in messages:
            if msg.get('role') == 'assistant':
                if current:
                    turns.append(current)
                    current = []
                # assistant message acts as boundary only; content is discarded
            else:
                current.append(msg)
        if current:
            turns.append(current)
        return turns

    def build_messages(self) -> Iterator[List[Messages]]:
        """Yield complete conversations as ``List[Messages]`` from the JSONL file.

        Each yielded item is a ``List[Messages]`` where every ``Messages``
        contains the delta for one turn.  The multi-turn benchmark runner
        extends the growing context with each delta and appends the model's
        real response after each turn.
        """
        max_turns = self.query_parameters.max_turns

        for line in self.dataset_line_by_line(self.query_parameters.dataset_path):
            line = line.strip()
            if not line:
                continue

            try:
                messages = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f'Skipping malformed JSON line: {e}')
                continue

            if not isinstance(messages, list) or not messages:
                logger.warning('Skipping line: expected a non-empty JSON array.')
                continue

            # Validate that every element has role and content fields
            if not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                logger.warning('Skipping line: each message must have "role" and "content" fields.')
                continue

            turns = self._split_into_turns(messages)

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
