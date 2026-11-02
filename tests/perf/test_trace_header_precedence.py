"""Trace headers fill missing HTTP fields without replacing CLI overrides."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.utils.body_meta import BODY_META_HEADERS, BODY_META_REQUEST_ID


@pytest.mark.parametrize('cli_name', ['Authorization', 'authorization', 'AUTHORIZATION'])
@pytest.mark.parametrize('trace_name', ['Authorization', 'authorization'])
def test_cli_header_precedence_is_case_insensitive(cli_name: str, trace_name: str) -> None:
    headers = {cli_name: 'Bearer cli-test-key'}
    trace_headers = {trace_name: 'Bearer trace-test-key', 'X-Trace-Only': 'trace'}
    body = {BODY_META_HEADERS: trace_headers, BODY_META_REQUEST_ID: 'request-1', 'model': 'test'}

    merged, request_id = ApiPluginBase.extract_body_meta(body, headers)

    assert merged == {cli_name: 'Bearer cli-test-key', 'X-Trace-Only': 'trace'}
    assert headers == {cli_name: 'Bearer cli-test-key'}
    assert trace_headers == {trace_name: 'Bearer trace-test-key', 'X-Trace-Only': 'trace'}
    assert request_id == 'request-1'
    assert body == {'model': 'test'}


def test_trace_headers_still_fill_missing_fields_and_drop_hop_by_hop_fields() -> None:
    body = {
        BODY_META_HEADERS: {
            'authorization': 'Bearer trace-test-key',
            'X-Route': 'trace-route',
            'hOsT': 'untrusted.example',
            'CONTENT-Length': '999',
            'Connection': 'close',
        }
    }

    merged, request_id = ApiPluginBase.extract_body_meta(body, {'Accept': 'application/json'})

    assert merged == {
        'authorization': 'Bearer trace-test-key',
        'X-Route': 'trace-route',
        'Accept': 'application/json',
    }
    assert request_id is None
    assert body == {}


@pytest.fixture
def provider() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    received: list[dict[str, Any]] = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append({
                'authorization': self.headers.get_all('Authorization'),
                'route': self.headers.get_all('X-Route'),
                'trace': self.headers.get_all('X-Trace-Only'),
                'body': json.loads(self.rfile.read(int(self.headers['Content-Length']))),
            })
            response = json.dumps({
                'object': 'chat.completion',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/chat/completions', received
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_replay_sends_only_cli_overrides(provider: tuple[str, list[dict[str, Any]]]) -> None:
    url, received = provider
    args = Arguments(
        model='test', api='openai', url=url, api_key='cli-test-key', headers={'X-Route': 'cli-route'}, stream=False
    )
    # The benchmark resolves the CLI's sweep list to one concurrency value.
    args.parallel = 1
    plugin = OpenaiPlugin(args)
    body = {
        'model': 'test',
        'messages': [],
        'stream': False,
        BODY_META_HEADERS: {
            'Authorization': 'Bearer trace-test-key',
            'authorization': 'Bearer trace-test-key',
            'X-Route': 'trace-route',
            'x-route': 'trace-route',
            'X-Trace-Only': 'trace',
        },
        BODY_META_REQUEST_ID: 'request-1',
    }

    async def send() -> None:
        async with AioHttpClient(args, plugin) as client:
            result = await client.post(body)
            assert result.success is True
            assert result.request_id == 'request-1'

    asyncio.run(send())

    assert received == [{
        'authorization': ['Bearer cli-test-key'],
        'route': ['cli-route'],
        'trace': ['trace'],
        'body': {'model': 'test', 'messages': [], 'stream': False},
    }]
