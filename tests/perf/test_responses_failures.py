"""Responses API application errors must remain failures after HTTP 200."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_responses_api import OpenAIResponsesPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator


def _response(status: str) -> dict:
    return {
        'id': 'resp_fixture',
        'object': 'response',
        'status': status,
        'error': {'code': 'server_error', 'message': 'upstream failed'} if status == 'failed' else None,
        'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None,
        'output': [{
            'type': 'message',
            'content': [{'type': 'output_text', 'text': 'partial answer'}],
        }],
        'usage': {
            'input_tokens': 7,
            'output_tokens': 3,
            'input_tokens_details': {'cached_tokens': 2},
        },
    }


@pytest.fixture
def provider() -> Iterator[str]:
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            mode, status = request['model'].split(':')
            response = _response(status)
            if mode == 'json':
                body = json.dumps(response).encode()
                content_type = 'application/json'
            else:
                terminal = (
                    {'type': 'error', 'code': 'server_error', 'message': 'upstream failed',
                     'param': None, 'sequence_number': 1}
                    if status == 'error'
                    else {'type': f'response.{status}', 'response': response, 'sequence_number': 1}
                )
                events = [
                    {'type': 'response.output_text.delta', 'delta': 'partial answer', 'sequence_number': 0},
                    terminal,
                ]
                body = ''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n' for e in events).encode()
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
        yield f'http://127.0.0.1:{server.server_port}/v1/responses'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def _request(plugin: OpenAIResponsesPlugin, url: str, model: str) -> BenchmarkData:
    async def send() -> BenchmarkData:
        async with aiohttp.ClientSession() as session:
            return await plugin.process_request(session, url, {}, {'model': model, 'input': 'answer'})

    return asyncio.run(send())


@pytest.mark.parametrize('model', ['json:failed', 'sse:error', 'sse:failed'])
def test_application_error_is_counted_as_failure(provider: str, model: str) -> None:
    plugin = OpenAIResponsesPlugin(Arguments(model=model, api='openai_responses'))

    output = _request(plugin, provider, model)

    assert output.success is False
    assert output.is_stream == model.startswith('sse:')
    assert output.generated_text == 'partial answer'
    assert 'upstream failed' in output.error
    assert output.response_messages
    assert output.completed_time >= output.start_time > 0

    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (0, 1)
    assert accumulator.total_completion_tokens == 0


@pytest.mark.parametrize('model', ['json:completed', 'sse:completed', 'json:incomplete', 'sse:incomplete'])
def test_completed_and_token_limited_responses_remain_successes(provider: str, model: str) -> None:
    plugin = OpenAIResponsesPlugin(Arguments(model=model, api='openai_responses'))

    output = _request(plugin, provider, model)

    assert output.success is True
    assert output.error is None
    assert output.generated_text == 'partial answer'
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (1, 0)
    assert (accumulator.total_prompt_tokens, accumulator.total_completion_tokens) == (7, 3)
