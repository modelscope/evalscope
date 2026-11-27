"""Tests for line_by_line JSON-scalar line classification.

Only JSON objects (complete request bodies) and arrays (messages format) are
JSON payloads. A line whose JSON parse yields a scalar (number, boolean,
null) is a plain-text prompt and must go through the plain-text path instead
of being forwarded as a request body.
"""

import json

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.datasets import base as base_mod
from evalscope.perf.plugin.datasets.line_by_line import LineByLineDatasetPlugin


class FakeTokenizer:
    """One token per character, matching the prefix-injection tests."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids)

    def __len__(self):
        return 256

    @property
    def all_special_ids(self):
        return []


def _build_dataset_plugin(tmp_path, lines, apply_chat_template=False, **kwargs):
    path = tmp_path / 'lines.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    suffix = 'chat/completions' if apply_chat_template else 'completions'
    args = Arguments(
        model='test-model',
        url=f'http://localhost:8080/v1/{suffix}',
        dataset='line_by_line',
        dataset_path=str(path),
        apply_chat_template=apply_chat_template,
        **kwargs,
    )
    return LineByLineDatasetPlugin(args)


class TestScalarLinesArePlainText:
    """Numbers, booleans and null lines are prompts, not request bodies."""

    def test_integer_line_is_plain_text_prompt(self, tmp_path):
        plugin = _build_dataset_plugin(tmp_path, ['42'])
        assert list(plugin.build_messages()) == ['42']

    def test_float_line_is_plain_text_prompt(self, tmp_path):
        plugin = _build_dataset_plugin(tmp_path, ['3.14'])
        assert list(plugin.build_messages()) == ['3.14']

    def test_negative_number_line_is_plain_text_prompt(self, tmp_path):
        plugin = _build_dataset_plugin(tmp_path, ['-7'])
        assert list(plugin.build_messages()) == ['-7']

    def test_boolean_line_is_plain_text_prompt(self, tmp_path):
        plugin = _build_dataset_plugin(tmp_path, ['true'])
        assert list(plugin.build_messages()) == ['true']

    def test_null_line_is_plain_text_prompt(self, tmp_path):
        plugin = _build_dataset_plugin(tmp_path, ['null'])
        assert list(plugin.build_messages()) == ['null']

    def test_scalar_prompt_wrapped_by_chat_template(self, tmp_path):
        plugin = _build_dataset_plugin(tmp_path, ['42'], apply_chat_template=True)
        assert list(plugin.build_messages()) == [[{'role': 'user', 'content': '42'}]]

    def test_quoted_json_string_still_unquoted(self, tmp_path):
        plugin = _build_dataset_plugin(tmp_path, ['"quoted prompt"'])
        assert list(plugin.build_messages()) == ['quoted prompt']

    def test_scalar_prompt_builds_valid_request(self, tmp_path):
        dataset_plugin = _build_dataset_plugin(tmp_path, ['42'])
        (prompt, ) = list(dataset_plugin.build_messages())
        api_plugin = OpenaiPlugin(dataset_plugin.query_parameters)
        result = api_plugin.build_request(prompt)
        assert result['prompt'] == '42'
        assert 'messages' not in result


class TestJsonPayloadsUnchanged:
    """Objects and arrays keep their JSON payload semantics."""

    def test_json_object_forwarded_verbatim(self, tmp_path):
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.6}
        plugin = _build_dataset_plugin(tmp_path, [json.dumps(body)])
        assert list(plugin.build_messages()) == [body]

    def test_json_array_forwarded_verbatim(self, tmp_path):
        messages = [{'role': 'user', 'content': 'hi'}]
        plugin = _build_dataset_plugin(tmp_path, [json.dumps(messages)])
        assert list(plugin.build_messages()) == [messages]

    def test_json_object_with_target_input_len_still_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: FakeTokenizer())
        plugin = _build_dataset_plugin(
            tmp_path,
            [json.dumps({'messages': [{'role': 'user', 'content': 'hi'}]})],
            tokenizer_path='fake',
            dataset_args={'target_input_len': 10},
        )
        with pytest.raises(ValueError, match='only support plain-text lines'):
            list(plugin.build_messages())


class TestScalarLengthControl:
    """Scalar prompts go through the plain-text path, so length control applies."""

    def test_scalar_prompt_capped_to_target_input_len(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base_mod, 'load_tokenizer', lambda path: FakeTokenizer())
        plugin = _build_dataset_plugin(
            tmp_path,
            ['123456789'],
            tokenizer_path='fake',
            dataset_args={'target_input_len': 5},
        )
        assert list(plugin.build_messages()) == ['12345']
