import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evalscope.perf.arguments import Arguments, parse_args
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.plugin.api.anthropic_api import AnthropicMessagesPlugin
from evalscope.perf.plugin.registry import ApiRegistry
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator
from evalscope.utils import get_secret_value


def test_request_conversion_url_and_authentication() -> None:
    args = Arguments(
        model='qwen3.8-flash',
        api='anthropic',
        url='https://dashscope.aliyuncs.com/apps/anthropic',
        api_key='fake-key',
        max_tokens=32,
        stop=['DONE'],
        stream=True,
    )
    plugin = ApiRegistry.get_class('anthropic')(args)

    body = plugin.build_request([
        {'role': 'system', 'content': 'Be concise'},
        {'role': 'user', 'content': 'Hello'},
        {'role': 'user', 'content': 'world'},
    ])

    assert args.url == 'https://dashscope.aliyuncs.com/apps/anthropic/v1/messages'
    assert get_secret_value(args.headers['x-api-key']) == 'fake-key'
    assert args.headers['anthropic-version'] == '2023-06-01'
    assert 'Authorization' not in args.headers
    assert body == {
        'model': 'qwen3.8-flash',
        'messages': [{'role': 'user', 'content': 'Hello\nworld'}],
        'system': 'Be concise',
        'max_tokens': 32,
        'stream': True,
        'stop_sequences': ['DONE'],
    }


@pytest.mark.parametrize(
    ('api', 'options', 'expected'),
    [
        ('anthropic', [], None),
        ('anthropic', ['--temperature', '0'], 0.0),
        ('openai', [], 0.0),
    ],
)
def test_cli_temperature_default_depends_on_protocol(
    monkeypatch: pytest.MonkeyPatch, api: str, options: list[str], expected: float | None
) -> None:
    monkeypatch.setattr('sys.argv', ['evalscope perf', '--api', api, '--model', 'test-model', *options])
    args = Arguments.from_args(parse_args())
    body = ApiRegistry.get_class(api)(args).build_request([{'role': 'user', 'content': 'hello'}])
    if expected is None:
        assert 'temperature' not in body
    else:
        assert body['temperature'] == expected


def test_complete_body_fields_take_precedence() -> None:
    args = Arguments(model='cli-model', api='anthropic', stream=True, extra_args={'thinking': {'type': 'enabled'}})
    plugin = AnthropicMessagesPlugin(args)
    original = {
        'model': 'body-model',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'max_tokens': 17,
        'stream': False,
        'temperature': 0.7,
        'thinking': {'type': 'disabled'},
    }
    body = plugin.build_request(original)

    assert body['model'] == 'body-model'
    assert body['max_tokens'] == 17
    assert body['stream'] is False
    assert body['temperature'] == 0.7
    assert body['thinking'] == {'type': 'disabled'}
    assert original == {
        'model': 'body-model',
        'messages': [{'role': 'user', 'content': 'hello'}],
        'max_tokens': 17,
        'stream': False,
        'temperature': 0.7,
        'thinking': {'type': 'disabled'},
    }


def test_custom_anthropic_version_header_is_case_insensitive() -> None:
    args = Arguments(model='test-model', api='anthropic', headers={'Anthropic-Version': 'custom-version'})
    assert args.headers['Anthropic-Version'] == 'custom-version'
    assert 'anthropic-version' not in args.headers


def test_stream_usage_is_combined_across_events() -> None:
    plugin = AnthropicMessagesPlugin(Arguments(model='test-model', api='anthropic'))
    responses = [
        {
            'type': 'message_start',
            'message': {
                'usage': {
                    'input_tokens': 7,
                    'output_tokens': 0,
                    'cache_read_input_tokens': 2,
                    'cache_creation_input_tokens': 3,
                }
            },
        },
        {'type': 'message_delta', 'usage': {'output_tokens': 3, 'cache_read_input_tokens': 2}},
    ]
    assert plugin.parse_responses(responses) == (12, 3)


def test_fully_cached_prompt_still_counts_input_tokens() -> None:
    plugin = AnthropicMessagesPlugin(Arguments(model='test-model', api='anthropic'))
    responses = [{
        'type': 'message',
        'usage': {'input_tokens': 0, 'output_tokens': 3, 'cache_read_input_tokens': 100},
    }]
    assert plugin.parse_responses(responses) == (100, 3)


@pytest.fixture
def provider() -> Iterator[tuple[str, dict]]:
    captured = {}

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            captured['headers'] = self.headers
            captured['body'] = request
            mode = request['model']
            if mode == 'json':
                payload = {
                    'type': 'message',
                    'content': [{'type': 'text', 'text': 'hello world'}],
                    'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 7, 'output_tokens': 3, 'cache_read_input_tokens': 2},
                }
                body = json.dumps(payload).encode()
                content_type = 'application/json'
            else:
                events = []
                if mode != 'missing_start':
                    events = [
                        {
                            'type': 'message_start',
                            'message': {
                                'type': 'message',
                                'content': [],
                                'usage': {'input_tokens': 7, 'output_tokens': 0, 'cache_read_input_tokens': 2},
                            },
                        },
                        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                    ]
                    if mode == 'tool':
                        events.extend([
                            {
                                'type': 'content_block_delta',
                                'index': 0,
                                'delta': {'type': 'input_json_delta', 'partial_json': '{"city":'},
                            },
                            {
                                'type': 'content_block_delta',
                                'index': 0,
                                'delta': {'type': 'input_json_delta', 'partial_json': '"Paris"}'},
                            },
                        ])
                    else:
                        events.extend([
                            {
                                'type': 'content_block_delta',
                                'index': 0,
                                'delta': {'type': 'text_delta', 'text': 'hello'},
                            },
                            {
                                'type': 'content_block_delta',
                                'index': 0,
                                'delta': {'type': 'text_delta', 'text': ' world'},
                            },
                        ])
                if mode == 'malformed':
                    events.append('{not-json')
                if mode == 'error':
                    events.append({'type': 'error', 'error': {'type': 'server_error', 'message': 'upstream failed'}})
                elif mode != 'incomplete':
                    events.append({
                        'type': 'message_delta',
                        'delta': {'stop_reason': 'end_turn'},
                        'usage': {'output_tokens': 3},
                    })
                    if mode != 'missing_stop':
                        events.append({'type': 'message_stop'})
                body = ''.join(
                    f'data: {event}\n\n'
                    if isinstance(event, str) else f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'
                    for event in events
                ).encode()
                content_type = 'text/event-stream'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/messages', captured
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def _request(args: Arguments, plugin: AnthropicMessagesPlugin) -> BenchmarkData:
    async def send() -> BenchmarkData:
        body = plugin.build_request([{'role': 'user', 'content': 'hello'}])
        async with AioHttpClient(args.model_copy(update={'parallel': 1}), plugin) as client:
            return await client.post(body)

    return asyncio.run(send())


@pytest.mark.parametrize('mode', ['json', 'sse'])
def test_successful_responses_feed_standard_metrics(provider: tuple[str, dict], mode: str) -> None:
    url, captured = provider
    args = Arguments(model=mode, api='anthropic', url=url, api_key='fake-key', stream=mode == 'sse')
    plugin = AnthropicMessagesPlugin(args)

    output = _request(args, plugin)

    assert output.success is True
    assert output.generated_text == 'hello world'
    assert (output.prompt_tokens, output.completion_tokens) == (9, 3)
    assert output.real_cached_tokens == 2
    assert captured['headers']['x-api-key'] == 'fake-key'
    assert captured['headers']['anthropic-version'] == '2023-06-01'
    assert captured['headers'].get('Authorization') is None
    assert captured['body']['stream'] is (mode == 'sse')
    if mode == 'sse':
        assert len(output.inter_chunk_latency) == 1
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (1, 0)
    assert (accumulator.total_prompt_tokens, accumulator.total_completion_tokens) == (9, 3)
    assert result.avg_cached_percent == pytest.approx(2 / 9 * 100)


def test_tool_argument_deltas_drive_timing(provider: tuple[str, dict]) -> None:
    url, _ = provider
    args = Arguments(model='tool', api='anthropic', url=url, stream=True)
    output = _request(args, AnthropicMessagesPlugin(args))

    assert output.success is True
    assert output.generated_text == ''
    assert output.first_chunk_latency > 0
    assert len(output.inter_chunk_latency) == 1


@pytest.mark.parametrize('mode', ['incomplete', 'error', 'malformed', 'missing_start', 'missing_stop'])
def test_incomplete_and_error_streams_fail(provider: tuple[str, dict], mode: str) -> None:
    url, _ = provider
    args = Arguments(model=mode, api='anthropic', url=url, stream=True)
    plugin = AnthropicMessagesPlugin(args)
    output = _request(args, plugin)

    assert output.success is False
    assert output.error
    if mode == 'error':
        assert 'upstream failed' in output.error
    if mode == 'missing_stop':
        assert 'message_stop' in output.error
        accumulator = MetricsAccumulator()
        accumulator.update(output, plugin)
        result = accumulator.to_result()
        assert (result.succeed_requests, result.failed_requests) == (0, 1)
