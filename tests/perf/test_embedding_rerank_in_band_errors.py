# Copyright (c) Alibaba, Inc. and its affiliates.
"""Embedding/rerank in-band errors reported inside an HTTP 200 JSON body must
remain failures.

vLLM and SGLang can send `{"error": {...}}` (or the flat `{"object": "error", ...}`
shape) with HTTP status 200 after the headers are already sent. The chat plugin
guards this (#1906/#1918); the embedding and rerank plugins used to set
`output.success = True` unconditionally on a 200, inflating the success rate.
"""

import asyncio
import json
import sys
import threading
import types
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Stub optional C-extension deps that evalscope.metrics pulls in eagerly, so
# the test can import the perf plugins without building them on Windows.
for _name in ('editdistance',):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)

import aiohttp
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_embedding_api import OpenaiEmbeddingPlugin
from evalscope.perf.plugin.api.openai_rerank_api import OpenaiRerankPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator

_ERROR = {'error': {'message': 'engine failed', 'type': 'InternalServerError', 'param': None, 'code': 500}}
_FLAT_ERROR = {'object': 'error', 'message': 'engine failed', 'type': 'InternalServerError', 'code': 500}

_EMBEDDING_BODIES = {
    'error_payload': _ERROR,
    'flat_error': _FLAT_ERROR,
    'ok': {
        'object': 'list',
        'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.1, 0.2, 0.3]}],
        'usage': {'prompt_tokens': 4, 'total_tokens': 4},
    },
}

_RERANK_BODIES = {
    'error_payload': _ERROR,
    'flat_error': _FLAT_ERROR,
    'ok': {
        'object': 'rerank',
        'results': [{'index': 0, 'relevance_score': 0.9, 'document': {'text': 'a'}}],
        'usage': {'prompt_tokens': 6, 'total_tokens': 6},
    },
}


def _make_provider(bodies: dict) -> type[BaseHTTPRequestHandler]:
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            model = self.path.rsplit('/', 1)[-1]
            body = json.dumps(bodies[model]).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Provider


@pytest.fixture
def embedding_provider() -> Iterator[str]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _make_provider(_EMBEDDING_BODIES))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.fixture
def rerank_provider() -> Iterator[str]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _make_provider(_RERANK_BODIES))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def _request(plugin, url: str, model: str, body: dict) -> BenchmarkData:
    async def send() -> BenchmarkData:
        async with aiohttp.ClientSession() as session:
            return await plugin.process_request(session, f'{url}/{model}', {}, body)

    return asyncio.run(send())


@pytest.mark.parametrize('model', ['error_payload', 'flat_error'])
def test_embedding_in_band_error_is_failure(embedding_provider: str, model: str) -> None:
    plugin = OpenaiEmbeddingPlugin(Arguments(model=model, api='openai_embedding'))

    output = _request(plugin, embedding_provider, model, {'model': model, 'input': 'hello'})

    assert output.success is False
    assert 'engine failed' in output.error
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (0, 1)


def test_embedding_ok_remains_success(embedding_provider: str) -> None:
    plugin = OpenaiEmbeddingPlugin(Arguments(model='ok', api='openai_embedding'))

    output = _request(plugin, embedding_provider, 'ok', {'model': 'ok', 'input': 'hello'})

    assert output.success is True
    assert output.error is None
    assert output.prompt_tokens == 4
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (1, 0)


@pytest.mark.parametrize('model', ['error_payload', 'flat_error'])
def test_rerank_in_band_error_is_failure(rerank_provider: str, model: str) -> None:
    plugin = OpenaiRerankPlugin(Arguments(model=model, api='openai_rerank'))

    output = _request(plugin, rerank_provider, model, {'model': model, 'query': 'q', 'documents': ['a', 'b']})

    assert output.success is False
    assert 'engine failed' in output.error
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (0, 1)


def test_rerank_ok_remains_success(rerank_provider: str) -> None:
    plugin = OpenaiRerankPlugin(Arguments(model='ok', api='openai_rerank'))

    output = _request(plugin, rerank_provider, 'ok', {'model': 'ok', 'query': 'q', 'documents': ['a', 'b']})

    assert output.success is True
    assert output.error is None
    assert output.prompt_tokens == 6
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (1, 0)
