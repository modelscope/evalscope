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


class _SpecialTokenTokenizer:
    """Configurable fake: an optional token it prepends (BOS) and an optional one it appends (EOS)."""

    chat_template = 'set'

    def __init__(self, template_bos: str, prepend: str = '<s>', append: str = '') -> None:
        self.template_bos = template_bos
        self.prepend = prepend
        self.append = append

    def apply_chat_template(self, conversation: List[Any], **kwargs: Any) -> str:
        return self.template_bos + ''.join(f'[{message["role"]}]{message["content"]}' for message in conversation)

    def _encode(self, text: str, add_special_tokens: bool) -> List[str]:
        pieces = [piece for piece in re.split(r'(<s>|</s>)', text) if piece]
        if add_special_tokens:
            if self.prepend:
                pieces.insert(0, self.prepend)
            if self.append:
                pieces.append(self.append)
        return pieces

    def decode(self, token_ids: List[str], **kwargs: Any) -> str:
        return ''.join(token_ids)

    def __call__(self, text: Any, add_special_tokens: bool = True, **kwargs: Any) -> dict:
        if isinstance(text, str):
            return {'input_ids': self._encode(text, add_special_tokens)}
        return {'input_ids': [self._encode(item, add_special_tokens) for item in text]}

    def batch_decode(self, *args: Any, **kwargs: Any) -> List[str]:
        return []


@pytest.mark.parametrize(
    ('template_bos', 'prepend', 'append', 'tokenizer_call_args', 'expected'),
    [
        # Template renders BOS, tokenizer also prepends it: the duplicate is stripped.
        ('<s>', '<s>', '', {}, '<s>[user]hi'),
        # Template renders no BOS: the tokenizer's prepended BOS is kept.
        ('', '<s>', '', {}, '<s>[user]hi'),
        # Tokenizer prepends BOS and appends EOS: only the duplicated BOS is stripped, EOS stays.
        ('<s>', '<s>', '</s>', {}, '<s>[user]hi</s>'),
        # Tokenizer only appends EOS: nothing is stripped, EOS stays.
        ('', '', '</s>', {}, '[user]hi</s>'),
        # An explicit add_special_tokens override is honored untouched.
        ('<s>', '<s>', '', {'add_special_tokens': True}, '<s><s>[user]hi'),
    ],
    ids=['bos-and-template-bos', 'bos-only', 'bos-and-eos', 'eos-only', 'explicit-add-special-tokens'],
)
def test_generate_special_token_prefix(
    monkeypatch: Any, template_bos: str, prepend: str, append: str, tokenizer_call_args: dict, expected: str
) -> None:
    """The rendered chat reaches the tokenizer with exactly the special tokens it should have."""
    encoded = []

    def fake_batched_generate(generate_input: Any) -> List[Any]:
        encoded.append(''.join(generate_input.tokenizer([generate_input.input])['input_ids'][0]))
        return [
            modelscope_module.GenerateOutput(
                output='ok', input_tokens=1, output_tokens=1, total_tokens=2, logprobs=None, time=0.1, stop_reason='stop'
            )
        ]

    monkeypatch.setattr(modelscope_module, 'batched_generate', fake_batched_generate)
    api = _make_api(_SpecialTokenTokenizer(template_bos, prepend=prepend, append=append))
    api.model_name = 'test-model'
    api.model = type('_Model', (), {'device': 'cpu', 'generate': lambda self, **kwargs: None})()
    api.tokenizer_call_args = tokenizer_call_args

    api.generate([ChatMessageUser(content='hi')], tools=[], tool_choice='none', config=GenerateConfig())

    assert encoded == [expected]
