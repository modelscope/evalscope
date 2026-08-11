"""Tests for the actionable error raised when a tokenizer has no usable chat template (issue #1548).

DeepSeek-V3.2 / V4 ship no Jinja chat template (the model card points at ``encoding`` scripts
instead) and base/pretrain checkpoints have none either, so ``apply_chat_template`` fails.  Perf
must explain what to do instead of leaking the raw transformers error.
"""

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets import base as base_mod
from evalscope.perf.plugin.datasets.line_by_line import LineByLineDatasetPlugin
from evalscope.perf.plugin.datasets.utils import tokenize_chat_messages

TEMPLATE_ERROR = 'cannot use chat template functions because tokenizer.chat_template is not set'
MESSAGES = [{'role': 'user', 'content': 'hi'}]


class NoTemplateTokenizer:
    """One token per character, and no chat template -- like a base checkpoint."""

    name_or_path = '/models/deepseek-v4-flash'

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        raise ValueError(TEMPLATE_ERROR)


class BareTokenizer:
    """Tokenizer that does not expose ``apply_chat_template`` at all."""

    name_or_path = 'bert-base-uncased'

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_error_reports_tokenizer_cause_and_options():
    with pytest.raises(ValueError) as excinfo:
        tokenize_chat_messages(NoTemplateTokenizer(), MESSAGES)
    message = str(excinfo.value)
    assert '/models/deepseek-v4-flash' in message
    assert TEMPLATE_ERROR in message
    assert 'drop `--tokenizer-path`' in message
    assert 'shares the vocabulary' in message
    assert '--no-apply-chat-template' in message
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_error_when_apply_chat_template_is_missing():
    with pytest.raises(ValueError) as excinfo:
        tokenize_chat_messages(BareTokenizer(), MESSAGES)
    assert 'bert-base-uncased' in str(excinfo.value)


def test_length_filter_reports_the_error(tmp_path, monkeypatch):
    """Reproduces #1548: a text dataset filtering prompts by token length on a chat endpoint."""
    monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: NoTemplateTokenizer())
    path = tmp_path / 'lines.txt'
    path.write_text('hello world', encoding='utf-8')
    args = Arguments(
        model='test-model',
        url='http://localhost:8080/v1/chat/completions',
        dataset='line_by_line',
        dataset_path=str(path),
        tokenizer_path='fake',
    )
    plugin = LineByLineDatasetPlugin(args)
    with pytest.raises(ValueError) as excinfo:
        list(plugin.build_messages())
    assert 'Failed to apply the chat template' in str(excinfo.value)
