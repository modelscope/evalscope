"""Exercise stream integrity through both providers' real HTTP clients."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import litellm
import pytest

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.models.litellm_compatible import LiteLLMAPI
from evalscope.models.openai_compatible import OpenAICompatibleAPI

HTTPProvider = tuple[str, list[bytes], list[dict[str, Any]]]


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        'id': 'completion-id',
        'created': 1,
        'model': 'gpt-4o-mini',
        'object': 'chat.completion.chunk',
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
    }


def _wire(chunks: list[dict[str, Any]], *, done: bool = True) -> bytes:
    return (
        ''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + ('data: [DONE]\n\n' if done else '')
    ).encode()


@pytest.fixture
def http_provider() -> Iterator[HTTPProvider]:
    responses: list[bytes] = []
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = responses[min(len(requests) - 1, len(responses) - 1)]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', responses, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _generate(provider: str, asynchronous: bool, base_url: str, *, n: int | None = None) -> ModelOutput:
    if provider == 'openai':
        api = OpenAICompatibleAPI(model_name='gpt-4o-mini', base_url=base_url, api_key='EMPTY', max_retries=0)
    else:
        api = LiteLLMAPI(model_name='openai/gpt-4o-mini', base_url=base_url, api_key='EMPTY')
    config = GenerateConfig(stream=True, retries=2, retry_interval=0, timeout=2, n=n)
    messages = [ChatMessageUser(content='question')]

    async def generate_async() -> ModelOutput:
        try:
            return await api.generate_async(messages, [], 'none', config)
        finally:
            await api.aclose()

    try:
        if asynchronous:
            return asyncio.run(generate_async())
        return api.generate(messages, [], 'none', config)
    finally:
        if isinstance(api, OpenAICompatibleAPI):
            api.client.close()


@pytest.mark.parametrize('provider', ['openai', 'litellm'])
@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('case', ['truncated', 'empty', 'role_only', 'usage_only'])
@pytest.mark.parametrize('exhausted', [False, True], ids=['recover', 'exhaust'])
def test_invalid_stream_retries_whole_request(
    http_provider: HTTPProvider, provider: str, asynchronous: bool, case: str, exhausted: bool
) -> None:
    base_url, responses, requests = http_provider
    invalid = {
        'truncated': _wire([_chunk({'content': 'discarded partial answer'})], done=False),
        'empty': _wire([]),
        'role_only': _wire([_chunk({'role': 'assistant'})]),
        'usage_only': _wire([{
            **_chunk({}), 'choices': [],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 0, 'total_tokens': 3},
        }]),
    }
    responses.append(invalid[case])
    if exhausted:
        with pytest.raises((RuntimeError, litellm.APIError), match='(?i)stream'):
            _generate(provider, asynchronous, base_url)
    else:
        responses.append(_wire([{
            **_chunk({'content': 'complete answer'}, 'stop'),
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
        }]))
        output = _generate(provider, asynchronous, base_url)
        assert output.completion == 'complete answer'
        assert output.stop_reason == 'stop'
        assert output.usage is not None and output.usage.total_tokens == 5
        assert output.perf_metrics is not None and output.perf_metrics.ttft is not None
        assert 0 <= output.perf_metrics.ttft <= output.time

    assert len(requests) == 2
    assert requests[0] == requests[1]


@pytest.mark.parametrize('provider', ['openai', 'litellm'])
@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('case,stop_reason', [
    ('text', 'stop'),
    ('no_done', 'stop'),
    ('empty_text', 'stop'),
    ('filtered', 'content_filter'),
    ('reasoning', 'stop'),
    ('tool_call', 'tool_calls'),
    ('trailing_metadata', 'max_tokens'),
    ('provider_reason', 'unknown'),
])
def test_explicitly_finished_stream_is_accepted(
    http_provider: HTTPProvider, provider: str, asynchronous: bool, case: str, stop_reason: str
) -> None:
    base_url, responses, requests = http_provider
    tool_call = {
        'index': 0, 'id': 'call-1', 'type': 'function',
        'function': {'name': 'lookup', 'arguments': '{"query":"ok"}'},
    }
    chunks = {
        'text': [_chunk({'content': 'answer'}, 'stop')],
        'no_done': [_chunk({'content': 'answer'}, 'stop')],
        'empty_text': [_chunk({}, 'stop')],
        'filtered': [_chunk({}, 'content_filter')],
        'reasoning': [_chunk({'reasoning_content': 'thinking'}, 'stop')],
        'tool_call': [_chunk({'tool_calls': [tool_call]}, 'tool_calls')],
        'trailing_metadata': [_chunk({'content': 'answer'}, 'length'), _chunk({})],
        'provider_reason': [_chunk({'content': 'answer'}, 'provider_specific')],
    }
    responses.append(_wire(chunks[case], done=case != 'no_done'))

    output = _generate(provider, asynchronous, base_url)

    assert len(requests) == 1
    assert len(output.choices) == 1
    assert output.stop_reason == stop_reason
    assert output.completion == (
        'answer' if case in {'text', 'no_done', 'trailing_metadata', 'provider_reason'} else ''
    )
    if case == 'reasoning':
        assert output.message.content[0].reasoning == 'thinking'
    if case == 'tool_call':
        assert output.message.tool_calls[0].function.name == 'lookup'
        assert output.message.tool_calls[0].function.arguments == {'query': 'ok'}
    if case == 'provider_reason':
        assert output.metadata == {'finish_reasons': {'0': 'provider_specific'}}


@pytest.mark.parametrize('provider', ['openai', 'litellm'])
@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('unfinished', [False, True], ids=['complete', 'unfinished'])
def test_multiple_choices_finish_independently(
    http_provider: HTTPProvider, provider: str, asynchronous: bool, unfinished: bool
) -> None:
    base_url, responses, requests = http_provider
    content = _chunk({'content': 'first'})
    content['choices'].append({'index': 1, 'delta': {'content': 'second'}, 'finish_reason': None})
    finish = _chunk({}, 'stop')
    if not unfinished:
        finish['choices'].append({'index': 1, 'delta': {}, 'finish_reason': 'length'})
    responses.append(_wire([content, finish]))

    if unfinished:
        with pytest.raises(RuntimeError, match='choices.*1'):
            _generate(provider, asynchronous, base_url, n=2)
        assert len(requests) == 2
    else:
        output = _generate(provider, asynchronous, base_url, n=2)
        assert len(requests) == 1
        assert [(choice.message.text, choice.stop_reason) for choice in output.choices] == [
            ('first', 'stop'), ('second', 'max_tokens'),
        ]
    assert requests[0]['n'] == 2
