import re
from typing import Any, List

import pytest
from jinja2 import Environment

from evalscope.api.messages import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.models import modelscope as modelscope_module
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


@pytest.mark.parametrize(
    ('message', 'field'),
    [(ChatMessageUser(content='hello'), 'metadata'), (ChatMessageAssistant(content='hello'), 'tool_calls')],
)
def test_ms_chat_preserves_none_for_legacy_templates(message: Any, field: str) -> None:
    class LegacyTokenizer:
        chat_template = '{{ messages[0].' + field + ' is none }}'

        def apply_chat_template(self, conversation: List[Any], **kwargs: Any) -> str:
            return Environment().from_string(self.chat_template).render(messages=conversation)

    assert _make_api(LegacyTokenizer()).ms_chat([message], tools=[]) == 'True'


class _BosTokenizer:
    """Prepends its BOS token by default, like the Llama 3, Gemma and Mistral tokenizers."""

    bos_token = '<s>'
    chat_template = 'set'

    def __init__(self, template_bos: str) -> None:
        self.template_bos = template_bos

    def apply_chat_template(self, conversation: List[Any], **kwargs: Any) -> str:
        return self.template_bos + ''.join(f'[{message["role"]}]{message["content"]}' for message in conversation)

    def _encode(self, text: str, add_special_tokens: bool) -> List[str]:
        pieces = [piece for piece in re.split(r'(<s>)', text) if piece]
        return [self.bos_token, *pieces] if add_special_tokens else pieces

    def __call__(self, text: Any, add_special_tokens: bool = True, **kwargs: Any) -> dict:
        if isinstance(text, str):
            return {'input_ids': self._encode(text, add_special_tokens)}
        return {'input_ids': [self._encode(item, add_special_tokens) for item in text]}

    def batch_decode(self, *args: Any, **kwargs: Any) -> List[str]:
        return []


@pytest.mark.parametrize('template_bos', ['<s>', ''])
def test_generate_sends_one_bos_token(monkeypatch: Any, template_bos: str) -> None:
    """A template that renders BOS gets no second one; a template without it keeps the tokenizer's."""
    encoded = []

    def fake_batched_generate(generate_input: Any) -> List[Any]:
        encoded.append(''.join(generate_input.tokenizer([generate_input.input])['input_ids'][0]))
        return [
            modelscope_module.GenerateOutput(
                output='ok', input_tokens=1, output_tokens=1, total_tokens=2, logprobs=None, time=0.1, stop_reason='stop'
            )
        ]

    monkeypatch.setattr(modelscope_module, 'batched_generate', fake_batched_generate)
    api = _make_api(_BosTokenizer(template_bos))
    api.model_name = 'test-model'
    api.model = type('_Model', (), {'device': 'cpu', 'generate': lambda self, **kwargs: None})()
    api.tokenizer_call_args = {}

    api.generate([ChatMessageUser(content='hi')], tools=[], tool_choice='none', config=GenerateConfig())

    assert encoded == ['<s>[user]hi']
