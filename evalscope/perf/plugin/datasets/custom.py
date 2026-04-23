import json
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
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

    The conversation is yielded as-is to the multi-turn benchmark runner.
    The runner discards all ``assistant`` messages from the dataset and
    accumulates only the model's **real** responses as conversation history,
    so the reference assistant content is never sent to the model.

    ``--max-turns`` optionally truncates long conversations to at most N user
    turns (and their preceding assistant replies).
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        if not query_parameters.dataset_path:
            raise ValueError(
                'custom_multi_turn requires --dataset-path to point to a local JSONL file. '
                'Each line must be a JSON array of OpenAI message dicts.'
            )

    def _truncate_to_max_turns(self, messages: List[Dict]) -> List[Dict]:
        """Truncate a conversation to at most max_turns user turns.

        Args:
            messages: Full conversation as a flat list of OpenAI message dicts.

        Returns:
            Truncated list containing at most ``max_turns`` user turns and
            their preceding/interleaved assistant messages.
        """
        max_turns = self.query_parameters.max_turns
        if max_turns is None:
            return messages

        truncated: List[Dict] = []
        user_turn_count = 0
        for msg in messages:
            if msg.get('role') == 'user':
                if user_turn_count >= max_turns:
                    break
                user_turn_count += 1
            truncated.append(msg)
        return truncated

    def build_messages(self) -> Iterator[List[Dict]]:
        """Yield complete conversations from the JSONL file.

        Each yielded item is a ``List[Dict]`` containing all messages of one
        conversation in OpenAI format.  The multi-turn benchmark runner will
        extract only user turns and use the model's real responses to build
        the growing context.
        """
        for line in self.dataset_line_by_line(self.query_parameters.dataset_path):
            line = line.strip()
            if not line:
                continue

            try:
                messages = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f'[custom_multi_turn] Skipping malformed JSON line: {e}')
                continue

            if not isinstance(messages, list) or not messages:
                logger.warning('[custom_multi_turn] Skipping line: expected a non-empty JSON array.')
                continue

            # Validate that every element has role and content fields
            if not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
                logger.warning('[custom_multi_turn] Skipping line: each message must have "role" and "content" fields.')
                continue

            # Apply max_turns truncation
            messages = self._truncate_to_max_turns(messages)

            # A valid multi-turn conversation needs at least one user turn
            user_turns = [m for m in messages if m.get('role') == 'user']
            if not user_turns:
                continue

            # Length filter: check the first user turn as a proxy
            first_user_content = user_turns[0]['content']
            if not isinstance(first_user_content, str):
                # Vision messages or other non-text content: skip length check
                yield messages
                continue

            is_valid, _ = self.check_prompt_length(first_user_content)
            if is_valid:
                yield messages


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
