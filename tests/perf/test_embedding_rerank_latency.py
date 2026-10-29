"""Embedding and rerank latency includes reading the complete response body."""

import asyncio
import json
import time

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.api.openai_embedding_api import OpenaiEmbeddingPlugin
from evalscope.perf.plugin.api.openai_rerank_api import OpenaiRerankPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator


@pytest.mark.parametrize('plugin_type', [OpenaiEmbeddingPlugin, OpenaiRerankPlugin])
@pytest.mark.parametrize('status', [200, 500])
@pytest.mark.parametrize('content_type', ['application/json', 'text/plain'])
def test_latency_includes_response_body(
    plugin_type: type[ApiPluginBase], status: int, content_type: str
) -> None:
    async def run() -> tuple[BenchmarkData, ApiPluginBase, float]:
        body_sent_at = 0.0
        payload = {
            'data': [{'embedding': [0.1, 0.2]}],
            'results': [{'index': 0, 'relevance_score': 0.9}],
            'usage': {'prompt_tokens': 10, 'total_tokens': 10},
        }

        async def handle(request: web.Request) -> web.StreamResponse:
            nonlocal body_sent_at
            await request.read()
            response = web.StreamResponse(status=status, headers={'Content-Type': content_type})
            await response.prepare(request)
            # Flush the headers separately, as a server/proxy may do while a
            # large embedding response is still being generated or transferred.
            await asyncio.sleep(0.05)
            body_sent_at = time.perf_counter()
            await response.write(json.dumps(payload).encode())
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post('/', handle)
        plugin = plugin_type(Arguments(model='test-model'))
        async with TestServer(app) as server:
            async with aiohttp.ClientSession() as session:
                output = await plugin.process_request(session, str(server.make_url('/')), {}, {})
        return output, plugin, body_sent_at

    output, plugin, body_sent_at = asyncio.run(run())

    assert output.success is (status == 200)
    assert output.completed_time >= body_sent_at
    assert output.query_latency == pytest.approx(output.completed_time - output.start_time)
    assert output.first_chunk_latency == output.query_latency

    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    assert accumulator.wall_time >= body_sent_at - output.start_time
