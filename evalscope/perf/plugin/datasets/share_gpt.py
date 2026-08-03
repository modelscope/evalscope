import json
import os
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.dataset_args import TextDatasetArgs, TextLengthArgs
from evalscope.perf.plugin.registry import register_dataset


class ShareGPTDatasetPluginBase(DatasetPluginBase):
    """Base class for ShareGPT dataset plugins.

    Data format per line (swift/sharegpt):
    {
        "conversation_id": "...",
        "category": "...",
        "conversation": [
            {"human": "...", "assistant": "..."},
            ...
        ]
    }

    Converts to OpenAI messages format (multi-turn), ending with a user turn:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
        {"role": "user", "content": "..."},   # last turn is always user
    ]

    Dataset: https://www.modelscope.cn/datasets/swift/sharegpt

    Length control (``target_input_len`` / ``prefix_file``) is applied over the
    whole conversation: conversations longer than the target are skipped and
    shorter ones are filled up by the prefix, so no turn is ever truncated.
    """

    # Subclasses must set this
    FILE_NAME: str = None

    args_schema = TextDatasetArgs

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        # `drop` keeps only inputs whose length already equals target_input_len.
        # For a multi-turn conversation that means the summed content of every
        # turn must hit the target exactly, which essentially never happens on
        # real data, so it would silently drop the whole dataset.
        args = self.dataset_args
        if (isinstance(args, TextLengthArgs) and args.target_input_len is not None and args.input_len_mode == 'drop'):
            raise ValueError(
                "`input_len_mode='drop'` is not supported for multi-turn ShareGPT datasets: a conversation's "
                'total content almost never equals target_input_len exactly, so every record would be dropped. '
                "Use `input_len_mode='cap'` (optionally with `prefix_file` to pad up to the target)."
            )

    def _convert_to_openai_messages(self, conversation: List[Dict]) -> List[Dict]:
        """Convert swift sharegpt conversation to OpenAI messages format.

        Each turn in the swift format is a dict with "human" and "assistant" keys.
        We interleave them into user/assistant messages and strip the trailing
        assistant turn so the conversation always ends with a user message.
        """
        messages = []
        for turn in conversation:
            human = turn.get('human', '').strip()
            assistant = turn.get('assistant', '').strip()
            if not human:
                continue
            messages.append({'role': 'user', 'content': human})
            if assistant:
                messages.append({'role': 'assistant', 'content': assistant})

        # Ensure the last message is from the user (model generates the response)
        if messages and messages[-1]['role'] == 'assistant':
            messages.pop()

        return messages

    def build_messages(self) -> Iterator[List[Dict]]:
        if not self.query_parameters.dataset_path or os.path.isdir(self.query_parameters.dataset_path):
            self.query_parameters.dataset_path = self.download_hub_file(
                dataset_id='swift/sharegpt', file_name=self.FILE_NAME
            )

        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            conversation = item.get('conversation', [])
            if not conversation:
                continue

            messages = self._convert_to_openai_messages(conversation)
            if not messages:
                continue

            # Length is measured over the whole conversation (see prepare_conversation).
            prepared = self.prepare_conversation(messages)
            if prepared is None:
                continue
            yield prepared


@register_dataset('share_gpt_zh')
class ShareGPTZhDatasetPlugin(ShareGPTDatasetPluginBase):
    """ShareGPT Chinese dataset plugin.
    File: common_zh_70k.jsonl (~70k Chinese conversations)
    Dataset: https://www.modelscope.cn/datasets/swift/sharegpt
    """

    FILE_NAME = 'common_zh_70k.jsonl'

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)


@register_dataset('share_gpt_en')
class ShareGPTEnDatasetPlugin(ShareGPTDatasetPluginBase):
    """ShareGPT English dataset plugin.
    File: common_en_70k.jsonl (~70k English conversations)
    Dataset: https://www.modelscope.cn/datasets/swift/sharegpt
    """

    FILE_NAME = 'common_en_70k.jsonl'

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
