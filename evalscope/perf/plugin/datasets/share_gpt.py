import json
import os
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
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
    """

    # Subclasses must set this
    FILE_NAME: str = None

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

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
        if not self.query_parameters.dataset_path:
            from modelscope import dataset_snapshot_download

            local_path = dataset_snapshot_download('swift/sharegpt', allow_patterns=[self.FILE_NAME])
            self.query_parameters.dataset_path = os.path.join(local_path, self.FILE_NAME)

        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            item = json.loads(item)
            conversation = item.get('conversation', [])
            if not conversation:
                continue

            messages = self._convert_to_openai_messages(conversation)
            if not messages:
                continue

            # Check prompt length using the last user turn content
            last_user_content = messages[-1]['content']
            is_valid, _ = self.check_prompt_length(last_user_content)
            if is_valid:
                yield messages


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
