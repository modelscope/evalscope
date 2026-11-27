"""Text token estimates must include Responses instructions and tool results."""

import json

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_responses_api import OpenAIResponsesPlugin


class _WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


@pytest.mark.parametrize(
    ('request_body', 'expected'),
    [
        ({'input': 'one two three'}, 3),
        ({'input': 'one', 'instructions': 'two three'}, 3),
        ({'input': 'one', 'instructions': None}, 1),
        ({'input': 'one', 'instructions': ''}, 1),
        ({'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'one two'}]}]}, 2),
        ({'input': [{'type': 'function_call_output', 'call_id': 'not counted', 'output': 'one two three'}]}, 3),
        ({'input': [{'type': 'function_call_output', 'call_id': 'not counted', 'output': ''}]}, 0),
        ({'input': [{'type': 'function_call_output', 'call_id': 'not counted', 'output': [
            {'type': 'input_text', 'text': 'one two'},
            {'type': 'input_image', 'image_url': 'not counted'},
            {'type': 'input_file', 'filename': 'not counted', 'file_data': 'not counted'},
            {'type': 'input_text', 'text': 'three'},
        ]}]}, 3),
        ({'instructions': 'one', 'input': [
            {'role': 'user', 'content': 'two'},
            {'type': 'function_call_output', 'call_id': 'not counted', 'output': 'three four'},
        ]}, 4),
    ],
)
def test_fallback_counts_request_text(request_body: dict, expected: int) -> None:
    plugin = OpenAIResponsesPlugin(Arguments(model='test', api='openai_responses'))
    plugin.tokenizer = _WordTokenizer()
    response = {'status': 'completed', 'output_text': 'answer text'}

    assert plugin.parse_responses([response], json.dumps(request_body)) == (expected, 2)


def test_server_usage_takes_precedence_over_fallback() -> None:
    plugin = OpenAIResponsesPlugin(Arguments(model='test', api='openai_responses'))
    plugin.tokenizer = _WordTokenizer()
    response = {'status': 'completed', 'usage': {'input_tokens': 42, 'output_tokens': 7}}
    request = {'instructions': 'one', 'input': [{'type': 'function_call_output', 'output': 'two'}]}

    assert plugin.parse_responses([response], json.dumps(request)) == (42, 7)


def test_missing_usage_still_requires_tokenizer() -> None:
    plugin = OpenAIResponsesPlugin(Arguments(model='test', api='openai_responses'))

    with pytest.raises(ValueError, match='no tokenizer'):
        plugin.parse_responses([{'status': 'completed', 'output_text': 'answer'}], '{"input": "question"}')
