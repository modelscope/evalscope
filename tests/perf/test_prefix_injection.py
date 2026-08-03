"""End-to-end tests for long-context prefix injection (issue #1524).

Every case drives the real pipeline (``Arguments`` -> plugin -> ``build_messages``)
with char-based fake tokenizers, so the budget arithmetic, message assembly and
config guardrails are all exercised through the public path without needing a
network or a real model.
"""

import json
import pytest
from pydantic import ValidationError

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets import base as base_mod
from evalscope.perf.plugin.datasets.line_by_line import LineByLineDatasetPlugin
from evalscope.perf.plugin.datasets.share_gpt import ShareGPTZhDatasetPlugin

PREFIX_CORPUS = 'PREFIXTEXTLONG'
TARGET = 10


class FakeTokenizer:
    """One token per character; fully reversible so token counts are exact."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids)

    def __len__(self):
        return 256

    @property
    def all_special_ids(self):
        return []


class DriftingTokenizer(FakeTokenizer):
    """Encodes ``~`` to a token that ``decode`` drops.

    This reproduces the real round-trip drift cause: prefix fitting decodes with
    ``skip_special_tokens=True``, so a naive slice comes back short and has to be
    topped up before it matches the budget.
    """

    DROPPED = 0

    def encode(self, text, add_special_tokens=False):
        return [self.DROPPED if c == '~' else ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids if not (skip_special_tokens and i == self.DROPPED))


def _tok_len(text: str, tokenizer=None) -> int:
    return len((tokenizer or FakeTokenizer()).encode(text))


def _write_prefix(tmp_path, text: str) -> str:
    path = tmp_path / 'prefix.txt'
    path.write_text(text, encoding='utf-8')
    return str(path)


def _build_line_plugin(tmp_path, monkeypatch, lines, tokenizer=None, apply_chat_template=True, **dataset_args):
    monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: tokenizer or FakeTokenizer())
    path = tmp_path / 'lines.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    url_suffix = 'chat/completions' if apply_chat_template else 'completions'
    args = Arguments(
        model='test-model',
        url=f'http://localhost:8080/v1/{url_suffix}',
        dataset='line_by_line',
        dataset_path=str(path),
        tokenizer_path='fake',
        apply_chat_template=apply_chat_template,
        dataset_args=dataset_args or None,
    )
    return LineByLineDatasetPlugin(args)


class TestSingleMessageInjection:
    """line_by_line: one user message per request."""

    def test_system_role_prepends_message(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['system', 'user']
        assert messages[1]['content'] == 'abc'
        assert sum(_tok_len(m['content']) for m in messages) == TARGET

    def test_user_role_prepends_inline(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
            prefix_role='user',
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['user']
        assert messages[0]['content'].endswith('abc')
        assert _tok_len(messages[0]['content']) == TARGET

    def test_plain_text_fallback_without_chat_template(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            apply_chat_template=False,
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
        )
        (prompt, ) = list(plugin.build_messages())
        assert isinstance(prompt, str)
        assert prompt.endswith('abc')
        assert _tok_len(prompt) == TARGET

    def test_short_corpus_is_tiled(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, 'PQR'),
        )
        (messages, ) = list(plugin.build_messages())
        assert messages[0]['content'] == 'PQRPQRP'
        assert sum(_tok_len(m['content']) for m in messages) == TARGET

    def test_tokenizer_round_trip_drift_is_compensated(self, tmp_path, monkeypatch):
        tokenizer = DriftingTokenizer()
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            tokenizer=tokenizer,
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, 'PQ~RS'),
        )
        (messages, ) = list(plugin.build_messages())
        prefix = messages[0]['content']
        # The dropped token is gone from the decoded prefix, yet the budget is
        # still met exactly because the shortfall was topped up.
        assert '~' not in prefix
        assert sum(_tok_len(m['content'], tokenizer) for m in messages) == TARGET


class TestConversationInjection:
    """share_gpt: multi-message conversation, budget measured over all turns."""

    def _build_plugin(self, tmp_path, monkeypatch, conversation=None, **dataset_args):
        monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: FakeTokenizer())
        conversation = conversation or [{'human': 'hi', 'assistant': 'yo'}, {'human': 'abc', 'assistant': ''}]
        path = tmp_path / 'sharegpt.jsonl'
        path.write_text(json.dumps({'conversation': conversation}) + '\n', encoding='utf-8')
        args = Arguments(
            model='test-model',
            url='http://localhost:8080/v1/chat/completions',
            dataset='share_gpt_zh',
            dataset_path=str(path),
            tokenizer_path='fake',
            apply_chat_template=True,
            dataset_args=dataset_args or None,
        )
        return ShareGPTZhDatasetPlugin(args)

    def test_system_prefix_fills_whole_conversation_budget(self, tmp_path, monkeypatch):
        plugin = self._build_plugin(
            tmp_path,
            monkeypatch,
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
        # History counts towards the budget: 'hi' + 'yo' + 'abc' = 7 tokens.
        assert _tok_len(messages[0]['content']) == TARGET - 7
        assert sum(_tok_len(m['content']) for m in messages) == TARGET
        assert messages[-1]['content'] == 'abc'

    def test_user_prefix_goes_to_first_message(self, tmp_path, monkeypatch):
        plugin = self._build_plugin(
            tmp_path,
            monkeypatch,
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
            prefix_role='user',
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['user', 'assistant', 'user']
        # 3 tokens of budget are left after the 7 tokens of conversation content.
        assert messages[0]['content'] == 'PREhi'
        assert sum(_tok_len(m['content']) for m in messages) == TARGET
        assert messages[-1]['content'] == 'abc'

    def test_conversation_longer_than_target_is_dropped(self, tmp_path, monkeypatch):
        # 'hi' + 'yo' + 12 chars = 16 tokens > TARGET; truncating a turn would
        # silently change the dialogue, so the whole record is skipped.
        plugin = self._build_plugin(
            tmp_path,
            monkeypatch,
            conversation=[{'human': 'hi', 'assistant': 'yo'}, {'human': 'abcdefghijkl', 'assistant': ''}],
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
        )
        assert list(plugin.build_messages()) == []

    def test_drop_mode_rejected_for_multi_turn(self, tmp_path, monkeypatch):
        # A conversation's summed content almost never equals the target exactly,
        # so `drop` would empty the dataset; the plugin refuses it at construction.
        with pytest.raises(ValueError, match='is not supported for multi-turn'):
            self._build_plugin(tmp_path, monkeypatch, target_input_len=TARGET, input_len_mode='drop')


class TestNoInjection:

    def test_prompt_filling_the_target_gets_no_prefix(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abcdefghijkl'],  # 12 tokens > target, leaving no prefix budget
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['user']
        assert _tok_len(messages[0]['content']) == TARGET

    def test_behavior_unchanged_without_prefix_file(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abcdefghij', 'abc'],
            target_input_len=5,
        )
        out = list(plugin.build_messages())
        assert out == [[{'role': 'user', 'content': 'abcde'}], [{'role': 'user', 'content': 'abc'}]]


class TestLineByLineJsonLines:
    """JSON lines bypass prepare_messages, so length control must fail fast."""

    MESSAGES_LINE = '[{"role": "user", "content": "abc"}]'
    BODY_LINE = '{"messages": [{"role": "user", "content": "abc"}], "max_tokens": 8}'

    @pytest.mark.parametrize('json_line', [MESSAGES_LINE, BODY_LINE], ids=['messages_array', 'request_body'])
    def test_json_line_rejected_with_length_control(self, tmp_path, monkeypatch, json_line):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc', json_line],
            target_input_len=TARGET,
            prefix_file=_write_prefix(tmp_path, PREFIX_CORPUS),
        )
        with pytest.raises(ValueError, match='only support plain-text lines'):
            list(plugin.build_messages())

    def test_json_line_rejected_with_target_input_len_only(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(tmp_path, monkeypatch, lines=[self.MESSAGES_LINE], target_input_len=TARGET)
        with pytest.raises(ValueError, match='only support plain-text lines'):
            list(plugin.build_messages())

    def test_json_line_passthrough_without_length_control(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(tmp_path, monkeypatch, lines=[self.MESSAGES_LINE, self.BODY_LINE])
        assert list(plugin.build_messages()) == [json.loads(self.MESSAGES_LINE), json.loads(self.BODY_LINE)]


@pytest.mark.parametrize(
    ('make_args', 'expected'),
    [
        (lambda p: {'target_input_len': TARGET, 'prefix_file': str(p / 'nope.txt')}, FileNotFoundError),
        (lambda p: {'target_input_len': TARGET, 'prefix_file': _write_prefix(p, '')}, ValueError),
        (lambda p: {'prefix_file': _write_prefix(p, PREFIX_CORPUS)}, ValidationError),
        (
            lambda p: {
                'target_input_len': TARGET,
                'prefix_file': _write_prefix(p, PREFIX_CORPUS),
                'input_len_mode': 'drop',
            },
            ValidationError,
        ),
    ],
    ids=['missing_file', 'empty_file', 'without_target_input_len', 'drop_mode'],
)
def test_invalid_prefix_config_is_rejected(tmp_path, monkeypatch, make_args, expected):
    with pytest.raises(expected):
        _build_line_plugin(tmp_path, monkeypatch, lines=['abc'], **make_args(tmp_path))
