import asyncio
import copy
import json
from pathlib import Path

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.benchmark import get_requests
from evalscope.perf.plugin.api.openai_responses_api import OpenAIResponsesPlugin


def _args(**kwargs: object) -> Arguments:
    return Arguments(model='cli-model', url='http://127.0.0.1:8000/v1/responses', **kwargs)


@pytest.mark.parametrize(
    ('field', 'value', 'cli'),
    [
        ('model', 'body-model', {}),
        ('max_output_tokens', 32, {}),
        ('stream', False, {}),
        ('stream', True, {'stream': False}),
        ('temperature', 0.7, {}),
        ('top_p', 0.8, {'top_p': 0.5}),
        ('temperature', None, {}),
    ],
)
def test_complete_responses_body_preserves_fields(field: str, value: object, cli: dict) -> None:
    plugin = OpenAIResponsesPlugin(_args(**cli))
    body = {'input': 'hello', field: value}
    original = copy.deepcopy(body)

    result = plugin.build_request(body)

    assert result[field] == value
    assert body == original


def test_complete_responses_body_fills_only_missing_extra_args() -> None:
    plugin = OpenAIResponsesPlugin(_args(extra_args={'temperature': 0.3, 'store': True, 'metadata': {'from': 'cli'}}))
    body = {'input': 'hello', 'store': False, 'metadata': {'from': 'body'}}
    original = copy.deepcopy(body)

    result = plugin.build_request(body)

    assert result['store'] is False
    assert result['metadata'] == {'from': 'body'}
    assert result['temperature'] == 0.0
    assert result['model'] == 'cli-model'
    assert result['max_output_tokens'] == 2048
    assert result['stream'] is True
    assert body == original


def test_complete_responses_body_adds_missing_extra_args() -> None:
    plugin = OpenAIResponsesPlugin(_args(extra_args={'store': False}))

    assert plugin.build_request({'input': 'hello'})['store'] is False


def test_complete_responses_messages_body_keeps_generation_fields() -> None:
    plugin = OpenAIResponsesPlugin(_args())
    messages = [{'role': 'user', 'content': 'hello'}]
    body = {'messages': messages, 'stream': False, 'max_output_tokens': 12}
    original = copy.deepcopy(body)

    result = plugin.build_request(body)

    assert result['input'] == messages
    assert 'messages' not in result
    assert result['stream'] is False
    assert result['max_output_tokens'] == 12
    assert body == original


@pytest.mark.parametrize('messages', ['hello', [{'role': 'user', 'content': 'hello'}]])
def test_responses_prompt_inputs_keep_cli_and_extra_args_precedence(messages: str | list) -> None:
    plugin = OpenAIResponsesPlugin(_args(temperature=0.6, extra_args={'temperature': 0.2}))

    result = plugin.build_request(messages)

    assert result['temperature'] == 0.2
    assert result['model'] == 'cli-model'
    assert result['max_output_tokens'] == 2048
    assert result['stream'] is True


@pytest.mark.parametrize('file_template', [False, True])
def test_responses_query_templates_keep_cli_override(tmp_path: Path, file_template: bool) -> None:
    template = json.dumps({'model': 'template-model', 'temperature': 0.9, 'stream': False, 'store': True})
    if file_template:
        path = tmp_path / 'template.json'
        path.write_text(template)
        template = f'@{path}'
    plugin = OpenAIResponsesPlugin(_args(query_template=template, extra_args={'store': False}))

    result = plugin.build_request({'input': 'hello'})

    assert result['model'] == 'cli-model'
    assert result['temperature'] == 0.0
    assert result['stream'] is True
    assert result['store'] is False


def test_responses_line_by_line_requests_keep_per_row_settings(tmp_path: Path) -> None:
    bodies = [
        {'input': 'first', 'model': 'first-model', 'max_output_tokens': 12, 'stream': False},
        {'input': 'second', 'model': 'second-model', 'max_output_tokens': 24, 'temperature': 0.7},
    ]
    path = tmp_path / 'requests.jsonl'
    path.write_text(''.join(json.dumps(body) + '\n' for body in bodies))
    args = _args(dataset='line_by_line', dataset_path=str(path), number=2, parallel=1, num_workers=1)
    plugin = OpenAIResponsesPlugin(args)

    async def collect() -> list:
        return [request async for request in get_requests(args, plugin)]

    requests = asyncio.run(collect())

    assert len(requests) == 2
    for (request, warmup), body in zip(requests, bodies):
        assert not warmup
        for field, value in body.items():
            assert request[field] == value
