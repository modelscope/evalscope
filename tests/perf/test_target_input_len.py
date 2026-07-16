"""Tests for target-input-length construction / truncation (issue #1483).

Unit-tests the ``truncate_text_to_token_len`` / ``fit_text_to_token_len``
helpers and their integration through ``DatasetPluginBase.prepare_prompt`` on a
real dataset plugin, using a lightweight char-based fake tokenizer (no network).
"""

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets import base as base_mod
from evalscope.perf.plugin.datasets.line_by_line import LineByLineDatasetPlugin
from evalscope.perf.plugin.datasets.utils import fit_text_to_token_len, truncate_text_to_token_len


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


# ---------------------------------------------------------------------------
# Unit: truncate / fit helpers
# ---------------------------------------------------------------------------


class TestTruncateHelper:

    def test_truncates_over_length(self):
        assert truncate_text_to_token_len('abcdefghij', 5, TOK) == 'abcde'

    def test_keeps_when_shorter(self):
        assert truncate_text_to_token_len('abc', 5, TOK) == 'abc'


class TestFitCapMode:

    def test_over_length_is_truncated(self):
        assert fit_text_to_token_len('abcdefghij', 5, 'cap', TOK) == 'abcde'

    def test_shorter_is_kept(self):
        assert fit_text_to_token_len('abc', 5, 'cap', TOK) == 'abc'


class TestFitDropMode:

    def test_over_length_is_truncated(self):
        out = fit_text_to_token_len('abcdefghij', 5, 'drop', TOK)
        assert out == 'abcde'
        assert _tok_len(out) == 5

    def test_shorter_is_dropped(self):
        assert fit_text_to_token_len('abc', 5, 'drop', TOK) is None


class TestFitInvalidMode:

    def test_unknown_mode_raises(self):
        try:
            fit_text_to_token_len('abc', 5, 'bogus', TOK)
            assert False, 'expected ValueError'
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Integration: prepare_prompt through line_by_line
# ---------------------------------------------------------------------------


def _build_plugin(tmp_path, monkeypatch, lines, **dataset_args):
    monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: FakeTokenizer())
    path = tmp_path / 'lines.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    args = Arguments(
        model='test-model',
        url='http://localhost:8080/v1/completions',
        dataset='line_by_line',
        dataset_path=str(path),
        tokenizer_path='fake',
        apply_chat_template=False,
        dataset_args=dataset_args or None,
    )
    return LineByLineDatasetPlugin(args)


class TestPreparePromptIntegration:

    def test_cap_truncates_and_keeps_short(self, tmp_path, monkeypatch):
        plugin = _build_plugin(
            tmp_path,
            monkeypatch,
            lines=['abcdefghij', 'abc'],
            target_input_len=5,
            input_len_mode='cap',
        )
        out = list(plugin.build_messages())
        assert out == ['abcde', 'abc']

    def test_drop_yields_only_exact_length(self, tmp_path, monkeypatch):
        plugin = _build_plugin(
            tmp_path,
            monkeypatch,
            lines=['abcdefghij', 'abc'],
            target_input_len=5,
            input_len_mode='drop',
        )
        out = list(plugin.build_messages())
        assert out == ['abcde']
        assert all(_tok_len(p) == 5 for p in out)

    def test_no_target_falls_back_to_length_filter(self, tmp_path, monkeypatch):
        # Without target_input_len, prompts pass through the min/max filter unchanged.
        plugin = _build_plugin(
            tmp_path,
            monkeypatch,
            lines=['abcdefghij', 'abc'],
        )
        out = list(plugin.build_messages())
        assert out == ['abcdefghij', 'abc']
