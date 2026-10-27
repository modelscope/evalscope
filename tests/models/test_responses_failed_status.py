"""A Responses API response with ``status == 'failed'`` must surface as a failure, not an empty answer.

Both the streaming and non-streaming ``responses.create`` paths can hand back a
``Response`` with ``status == 'failed'`` (and an ``error`` describing why) after HTTP
200. Downstream code otherwise reads it as an empty assistant reply, so a failed
request is scored as a blank answer — the same family as #1906/#1918 on the Chat
Completions path. See modelscope/evalscope#1906, #1918.
"""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.models.openai_responses import OpenAIResponsesAPI
from evalscope.models.utils.openai import EmptyCompletionError


def _response(status: str) -> dict:
    return {
        'id': 'resp_fixture',
        'object': 'response',
        'created_at': 1,
        'model': 'fixture',
        'status': status,
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'error': {'code': 'server_error', 'message': 'upstream failed'} if status == 'failed' else None,
        'incomplete_details': None,
        'output': [],
        'usage': {
            'input_tokens': 7,
            'output_tokens': 0,
            'total_tokens': 7,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens_details': {'reasoning_tokens': 0},
        },
    }


def _sse(event: dict) -> bytes:
    return f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'.encode()


@pytest.fixture
def provider() -> Iterator[tuple[str, list[dict]]]:
    requests: list[dict] = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(request)
            response = _response('failed')
            if request.get('stream'):
                events = [
                    {'type': 'response.output_text.delta', 'delta': '', 'sequence_number': 0},
                    {'type': 'response.failed', 'sequence_number': 1, 'response': response},
                ]
                body = b''.join(_sse(e) for e in events)
                content_type = 'text/event-stream'
            else:
                body = json.dumps(response).encode()
                content_type = 'application/json'
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
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def _generate(api: OpenAIResponsesAPI, asynchronous: bool, stream: bool, retries: int = 2) -> None:
    kwargs = {
        'input': [ChatMessageUser(content='answer')],
        'tools': [],
        'tool_choice': 'none',
        'config': GenerateConfig(retries=retries, retry_interval=0, stream=stream),
    }

    async def generate_async() -> None:
        try:
            await api.generate_async(**kwargs)
        finally:
            await api.aclose()

    try:
        if asynchronous:
            asyncio.run(generate_async())
        else:
            api.generate(**kwargs)
    finally:
        api.client.close()


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('stream', [False, True], ids=['non-stream', 'stream'])
def test_failed_response_raises_instead_of_returning_empty(
    provider: tuple[str, list[dict]], asynchronous: bool, stream: bool,
) -> None:
    base_url, requests = provider
    api = OpenAIResponsesAPI(model_name='failed', base_url=base_url, api_key='EMPTY', max_retries=0)

    # A failed response carries the error detail into the exception message so the
    # failure is diagnosable, not a silent blank answer.
    with pytest.raises(EmptyCompletionError, match='status=failed'):
        _generate(api, asynchronous, stream)

    # failed responses go through the same retry path as #1918: the request is retried
    # before the sample is recorded as an error.
    assert len(requests) == 2


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('stream', [False, True], ids=['non-stream', 'stream'])
def test_failed_response_error_detail_in_message(
    provider: tuple[str, list[dict]], asynchronous: bool, stream: bool,
) -> None:
    base_url, _requests = provider
    api = OpenAIResponsesAPI(model_name='failed', base_url=base_url, api_key='EMPTY', max_retries=0)

    with pytest.raises(EmptyCompletionError, match='server_error') as exc_info:
        _generate(api, asynchronous, stream)

    assert 'upstream failed' in str(exc_info.value)
