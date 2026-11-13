from typing import Any, List

from evalscope.api.messages import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.models.modelscope import ModelScopeAPI


class _DictOnlyTokenizer:
    """Mimics `apply_chat_template` in transformers>=5.23, which calls `.items()` on every message."""

    chat_template = '{{ messages }}'

    def __init__(self) -> None:
        self.conversation: List[dict] = []

    def apply_chat_template(self, conversation: List[Any], **kwargs: Any) -> str:
        self.conversation = [{k: v for k, v in message.items() if v is not None} for message in conversation]
        return '|'.join(f'{message["role"]}:{message["content"]}' for message in self.conversation)


def _make_api(tokenizer: Any) -> ModelScopeAPI:
    """Build a ModelScopeAPI without loading any weights."""
    api = object.__new__(ModelScopeAPI)
    api.tokenizer = tokenizer
    api.enable_thinking = None
    return api


def test_ms_chat_passes_plain_dict_messages_to_chat_template() -> None:
    tokenizer = _DictOnlyTokenizer()
    messages = [
        ChatMessageSystem(content='Be brief.'),
        ChatMessageUser(content='Weather in Paris?'),
        ChatMessageAssistant(
            content='',
            tool_calls=[ToolCall(id='call_1', function=ToolFunction(name='get_weather', arguments={'city': 'Paris'}))],
        ),
        ChatMessageTool(content='18C', tool_call_id='call_1', function='get_weather'),
    ]

    chat = _make_api(tokenizer).ms_chat(messages, tools=[])

    assert chat == 'system:Be brief.|user:Weather in Paris?|assistant:|tool:18C'
    assistant = tokenizer.conversation[2]
    assert assistant['tool_calls'][0]['function'] == {'name': 'get_weather', 'arguments': {'city': 'Paris'}}
    assert tokenizer.conversation[3]['tool_call_id'] == 'call_1'
    # Unset optional fields are left out, as chat templates test for missing keys.
    assert 'tool_call_id' not in tokenizer.conversation[1]
    assert 'metadata' not in tokenizer.conversation[1]
    # The caller's messages are not modified.
    assert isinstance(messages[0], ChatMessageSystem)
