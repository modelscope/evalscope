"""Retry Responses requests whose SSE body is interrupted after HTTP 200."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from openai import AuthenticationError

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig, ModelOutput
from evalscope.models.openai_responses import OpenAIResponsesAPI


def _response() -> dict:
    return {
        'id': 'resp_complete',
        'object': 'response',
        'created_at': 1,
        'model': 'fixture',
        'status': 'completed',
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'error': None,
        'incomplete_details': None,
        'output': [{
            'type': 'message',
            'id': 'msg_complete',
            'role': 'assistant',
            'status': 'completed',
            'content': [{'type': 'output_text', 'text': 'complete answer', 'annotations': []}],
        }],
        'usage': {
            'input_tokens': 7,
            'output_tokens': 3,
            'total_tokens': 10,
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
            model = request['model']
            attempt = len(requests)
            if model == 'unauthorized':
                body = json.dumps({'error': {'message': 'invalid key', 'type': 'invalid_request_error'}}).encode()
                self.send_response(401)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            interrupted = attempt == 1 and model != 'healthy' or model.startswith('always-')
            delta = {
                'type': 'response.output_text.delta',
                'sequence_number': 0,
                'item_id': 'msg_partial' if interrupted else 'msg_complete',
                'output_index': 0,
                'content_index': 0,
                'delta': 'discarded partial answer' if interrupted else 'complete answer',
            }
            body = _sse(delta)
            if not interrupted:
                body += _sse({'type': 'response.completed', 'sequence_number': 1, 'response': _response()})
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            # A longer advertised body forces a transport error after the first delta.
            size = len(body) + 100 if interrupted and model.endswith('disconnect') else len(body)
            self.send_header('Content-Length', str(size))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = True

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


def _generate(api: OpenAIResponsesAPI, asynchronous: bool, retries: int = 2) -> ModelOutput:
    kwargs = {
        'input': [ChatMessageUser(content='answer')],
        'tools': [],
        'tool_choice': 'none',
        'config': GenerateConfig(retries=retries, retry_interval=0, stream=True),
    }

    async def generate_async() -> ModelOutput:
        try:
            return await api.generate_async(**kwargs)
        finally:
            await api.aclose()

    try:
        return asyncio.run(generate_async()) if asynchronous else api.generate(**kwargs)
    finally:
        api.client.close()


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('failure', ['disconnect', 'missing-terminal'])
def test_retries_interrupted_response_stream(provider: tuple[str, list[dict]], asynchronous: bool, failure: str) -> None:
    base_url, requests = provider
    api = OpenAIResponsesAPI(model_name=failure, base_url=base_url, api_key='EMPTY', max_retries=0)

    output = _generate(api, asynchronous)

    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert output.message.text == 'complete answer'
    assert output.usage is not None
    assert (output.usage.input_tokens, output.usage.output_tokens) == (7, 3)
    assert output.message.perf_metrics is not None
    assert 0 <= output.message.perf_metrics.ttft <= output.time


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('failure,error', [('disconnect', httpx.RemoteProtocolError), ('missing-terminal', ValueError)])
def test_response_stream_retry_limit(
    provider: tuple[str, list[dict]], asynchronous: bool, failure: str, error: type[Exception],
) -> None:
    base_url, requests = provider
    api = OpenAIResponsesAPI(model_name=f'always-{failure}', base_url=base_url, api_key='EMPTY', max_retries=0)

    with pytest.raises(error):
        _generate(api, asynchronous)

    assert len(requests) == 2


@pytest.mark.parametrize('asynchronous', [False, True], ids=['sync', 'async'])
@pytest.mark.parametrize('model', ['healthy', 'unauthorized'])
def test_response_stream_does_not_retry_success_or_client_errors(
    provider: tuple[str, list[dict]], asynchronous: bool, model: str,
) -> None:
    base_url, requests = provider
    api = OpenAIResponsesAPI(model_name=model, base_url=base_url, api_key='EMPTY', max_retries=0)

    if model == 'unauthorized':
        with pytest.raises(AuthenticationError, match='invalid key'):
            _generate(api, asynchronous)
    else:
        output = _generate(api, asynchronous)
        assert output.message.text == 'complete answer'

    assert len(requests) == 1
