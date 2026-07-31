"""Tests for long-context prefix injection (issue #1524).

Unit-tests the shared ``converge_to_token_len`` skeleton and both of its
wrappers (``fit_prefix_to_budget`` / ``gen_prompt_decode_to_target_len``), the
``prefix_file`` / ``prefix_role`` schema validation, and the end-to-end
injection through ``line_by_line`` / ``share_gpt`` plugins, using lightweight
char-based fake tokenizers (no network).
"""

import json
import numpy as np
import pytest
from pydantic import ValidationError

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets import base as base_mod
from evalscope.perf.plugin.datasets.dataset_args import BaseDatasetArgs, TextDatasetArgs
from evalscope.perf.plugin.datasets.line_by_line import LineByLineDatasetPlugin
from evalscope.perf.plugin.datasets.share_gpt import ShareGPTZhDatasetPlugin
from evalscope.perf.plugin.datasets.utils import (
    converge_to_token_len,
    fit_prefix_to_budget,
    gen_prompt_decode_to_target_len,
)


class FakeTokenizer:
    """One token per character; fully reversible so token lengths are exact."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids)

    def __len__(self):
        return 256

    @property
    def all_special_ids(self):
        return []


TOK = FakeTokenizer()


def _tok_len(text: str) -> int:
    return len(TOK.encode(text))


def _ids(text: str):
    return TOK.encode(text)


# ---------------------------------------------------------------------------
# Unit: fit_prefix_to_budget
# ---------------------------------------------------------------------------


class TestFitPrefixToBudget:

    def test_exact_slice(self):
        out = fit_prefix_to_budget(_ids('PREFIXTEXT'), 5, TOK)
        assert out == 'PREFI'
        assert _tok_len(out) == 5

    def test_tiling_when_prefix_too_short(self):
        out = fit_prefix_to_budget(_ids('PQR'), 8, TOK)
        assert out == 'PQRPQRPQ'
        assert _tok_len(out) == 8

    def test_zero_or_negative_budget_returns_empty(self):
        assert fit_prefix_to_budget(_ids('PQR'), 0, TOK) == ''
        assert fit_prefix_to_budget(_ids('PQR'), -3, TOK) == ''

    def test_empty_prefix_raises(self):
        with pytest.raises(ValueError):
            fit_prefix_to_budget([], 5, TOK)


# ---------------------------------------------------------------------------
# Unit: converge_to_token_len (shared decode/re-encode skeleton)
# ---------------------------------------------------------------------------


class MergeTokenizer(FakeTokenizer):
    """Encodes the pair 'ab' into a single token, so re-encoding shrinks length."""

    AB = 1000

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] == 'ab':
                ids.append(self.AB)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return ''.join('ab' if i == self.AB else chr(i) for i in ids)


class SplitTokenizer(FakeTokenizer):
    """Encodes 'A' into two tokens, so re-encoding always inflates length."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for c in text:
            ids.extend([ord(c), ord(c)] if c == 'A' else [ord(c)])
        return ids


class TestConvergeToTokenLen:

    def test_returns_immediately_when_already_exact(self):
        calls = []
        text, ids, mismatch = converge_to_token_len(TOK, _ids('abc'), 3, fill=lambda n: calls.append(n) or [])
        assert (text, mismatch, calls) == ('abc', 0, [])
        assert len(ids) == 3

    def test_fills_shortfall_from_callback(self):
        # 'abcd' re-encodes to 3 tokens (ab merged), so one filler token is needed.
        text, ids, mismatch = converge_to_token_len(MergeTokenizer(), _ids('abcd'), 4, fill=lambda n: [ord('e')] * n)
        assert (text, mismatch) == ('abcde', 0)
        assert len(ids) == 4

    def test_reports_consistent_mismatch_when_not_converging(self):
        # 'A' always re-encodes to 2 tokens: unreachable target, must not hang.
        tok = SplitTokenizer()
        text, ids, mismatch = converge_to_token_len(tok, _ids('A'), 1, fill=lambda n: [], max_retry=3)
        assert mismatch == len(ids) - 1 == 1
        assert ids == tok.encode(text)


class TestGenPromptDecodeToTargetLen:
    """Regression cover for the random-dataset wrapper over the shared skeleton."""

    def test_fills_from_allowed_tokens(self):
        prompt, ids, mismatch = gen_prompt_decode_to_target_len(
            tokenizer=MergeTokenizer(),
            token_sequence=_ids('abcd'),
            target_token_len=4,
            allowed_tokens=np.array([ord('e')]),
        )
        assert (prompt, mismatch) == ('abcde', 0)
        assert len(ids) == 4

    def test_reports_mismatch_when_target_unreachable(self):
        _, ids, mismatch = gen_prompt_decode_to_target_len(
            tokenizer=SplitTokenizer(),
            token_sequence=_ids('A'),
            target_token_len=1,
            allowed_tokens=np.array([ord('e')]),
        )
        assert mismatch == len(ids) - 1 == 1


# ---------------------------------------------------------------------------
# Unit: schema validation
# ---------------------------------------------------------------------------


class TestPrefixSchema:

    def test_prefix_file_requires_target_input_len(self):
        with pytest.raises(ValidationError):
            TextDatasetArgs(prefix_file='/tmp/prefix.txt')

    def test_invalid_prefix_role_rejected(self):
        with pytest.raises(ValidationError):
            TextDatasetArgs(target_input_len=10, prefix_file='/tmp/prefix.txt', prefix_role='tool')

    def test_prefix_file_rejected_on_base_schema(self):
        # Datasets without TextLengthArgs (e.g. random) reject prefix_file via extra='forbid'.
        with pytest.raises(ValidationError):
            BaseDatasetArgs(prefix_file='/tmp/prefix.txt')


# ---------------------------------------------------------------------------
# Integration: line_by_line
# ---------------------------------------------------------------------------


def _write_prefix(tmp_path, text: str) -> str:
    path = tmp_path / 'prefix.txt'
    path.write_text(text, encoding='utf-8')
    return str(path)


def _build_line_plugin(tmp_path, monkeypatch, lines, apply_chat_template, **dataset_args):
    monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: FakeTokenizer())
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


class TestLineByLinePrefixInjection:

    def test_system_role_injection_hits_target(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            apply_chat_template=True,
            target_input_len=10,
            prefix_file=_write_prefix(tmp_path, 'PREFIXTEXTLONG'),
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['system', 'user']
        assert messages[1]['content'] == 'abc'
        total = sum(_tok_len(m['content']) for m in messages)
        assert total == 10

    def test_user_role_injection_hits_target(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            apply_chat_template=True,
            target_input_len=10,
            prefix_file=_write_prefix(tmp_path, 'PREFIXTEXTLONG'),
            prefix_role='user',
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['user']
        assert messages[0]['content'].endswith('abc')
        assert _tok_len(messages[0]['content']) == 10

    def test_no_chat_template_falls_back_to_text_concat(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            apply_chat_template=False,
            target_input_len=10,
            prefix_file=_write_prefix(tmp_path, 'PREFIXTEXTLONG'),
        )
        (prompt, ) = list(plugin.build_messages())
        assert isinstance(prompt, str)
        assert prompt.endswith('abc')
        assert _tok_len(prompt) == 10

    def test_over_length_prompt_gets_no_prefix(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abcdefghijkl'],  # 12 tokens > target 10
            apply_chat_template=True,
            target_input_len=10,
            prefix_file=_write_prefix(tmp_path, 'PREFIXTEXTLONG'),
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['user']
        assert _tok_len(messages[0]['content']) == 10

    def test_short_prefix_is_tiled_to_target(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abc'],
            apply_chat_template=True,
            target_input_len=10,
            prefix_file=_write_prefix(tmp_path, 'PQR'),
        )
        (messages, ) = list(plugin.build_messages())
        assert messages[0]['content'] == 'PQRPQRP'
        total = sum(_tok_len(m['content']) for m in messages)
        assert total == 10

    def test_missing_prefix_file_raises(self, tmp_path, monkeypatch):
        with pytest.raises(FileNotFoundError):
            _build_line_plugin(
                tmp_path,
                monkeypatch,
                lines=['abc'],
                apply_chat_template=True,
                target_input_len=10,
                prefix_file=str(tmp_path / 'nope.txt'),
            )

    def test_no_prefix_keeps_existing_behavior(self, tmp_path, monkeypatch):
        plugin = _build_line_plugin(
            tmp_path,
            monkeypatch,
            lines=['abcdefghij', 'abc'],
            apply_chat_template=True,
            target_input_len=5,
        )
        out = list(plugin.build_messages())
        assert out == [[{'role': 'user', 'content': 'abcde'}], [{'role': 'user', 'content': 'abc'}]]


# ---------------------------------------------------------------------------
# Integration: share_gpt (multi-message conversation)
# ---------------------------------------------------------------------------


class TestShareGPTPrefixInjection:

    def _build_plugin(self, tmp_path, monkeypatch, **dataset_args):
        monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: FakeTokenizer())
        record = {'conversation': [{'human': 'hi', 'assistant': 'yo'}, {'human': 'abc', 'assistant': ''}]}
        path = tmp_path / 'sharegpt.jsonl'
        path.write_text(json.dumps(record) + '\n', encoding='utf-8')
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

    def test_system_prefix_prepended_to_conversation(self, tmp_path, monkeypatch):
        plugin = self._build_plugin(
            tmp_path,
            monkeypatch,
            target_input_len=10,
            prefix_file=_write_prefix(tmp_path, 'PREFIXTEXTLONG'),
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
        # Budget is measured against the last user turn only ('abc' -> 3 tokens).
        assert _tok_len(messages[0]['content']) == 7
        assert messages[-1]['content'] == 'abc'

    def test_user_prefix_goes_to_first_message(self, tmp_path, monkeypatch):
        plugin = self._build_plugin(
            tmp_path,
            monkeypatch,
            target_input_len=10,
            prefix_file=_write_prefix(tmp_path, 'PREFIXTEXTLONG'),
            prefix_role='user',
        )
        (messages, ) = list(plugin.build_messages())
        assert [m['role'] for m in messages] == ['user', 'assistant', 'user']
        assert messages[0]['content'].endswith('hi')
        assert messages[0]['content'].startswith('PREFIX')
        assert messages[-1]['content'] == 'abc'
