"""Chat Completions errors reported inside an HTTP 200 SSE stream must remain failures."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator

# vLLM and SGLang send `{"error": {...}}`; older vLLM sends the flat `{"object": "error", ...}` shape.
_ERROR = {'error': {'message': 'engine failed', 'type': 'InternalServerError', 'param': None, 'code': 500}}
_FLAT_ERROR = {'object': 'error', 'message': 'engine failed', 'type': 'InternalServerError', 'code': 500}


def _chunk(text: str) -> dict:
    return {'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': text}}]}


_STREAMS = {
    'error_first': [_ERROR],
    'flat_error_first': [_FLAT_ERROR],
    'error_after_output': [_chunk('par'), _chunk('tial'), _ERROR],
    'completed': [
        _chunk('par'),
        _chunk('tial'),
        {'object': 'chat.completion.chunk', 'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 2}},
    ],
}


@pytest.fixture
def provider() -> Iterator[str]:
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            events = _STREAMS[request['model']]
            body = (''.join(f'data: {json.dumps(e)}\n\n' for e in events) + 'data: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def _request(plugin: OpenaiPlugin, url: str, model: str) -> BenchmarkData:
    async def send() -> BenchmarkData:
        async with aiohttp.ClientSession() as session:
            return await plugin.process_request(session, url, {}, {'model': model, 'stream': True, 'messages': []})

    return asyncio.run(send())


@pytest.mark.parametrize('model', ['error_first', 'flat_error_first', 'error_after_output'])
def test_in_band_stream_error_is_counted_as_failure(provider: str, model: str) -> None:
    plugin = OpenaiPlugin(Arguments(model=model, api='openai'))

    output = _request(plugin, provider, model)

    assert output.success is False
    assert output.is_stream is True
    assert 'engine failed' in output.error
    assert output.completed_time >= output.start_time > 0

    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (0, 1)
    assert accumulator.total_first_chunk_latency == 0
    assert accumulator.total_completion_tokens == 0


def test_completed_stream_remains_success(provider: str) -> None:
    plugin = OpenaiPlugin(Arguments(model='completed', api='openai'))

    output = _request(plugin, provider, 'completed')

    assert output.success is True
    assert output.error is None
    assert output.generated_text == 'partial'
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (1, 0)
    assert (accumulator.total_prompt_tokens, accumulator.total_completion_tokens) == (7, 2)
