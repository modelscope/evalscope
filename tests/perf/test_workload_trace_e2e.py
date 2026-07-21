"""End-to-end tests for workload_trace: real HTTP server + full dispatch pipeline."""

import asyncio
import json
import os
import time
from aiohttp import web

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.core.strategies.open_loop import OpenLoopStrategy
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.datasets.workload_trace import WorkloadTraceDatasetPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.perf.utils.body_meta import BODY_META_PREFIX


class EchoServer:
    """In-process aiohttp server that records arrival times and request bodies."""

    def __init__(self):
        self.arrivals: list[float] = []
        self.bodies: list[dict] = []
        self.headers_log: list[dict] = []
        self._runner = None
        self.port = None

    async def _handle(self, request: web.Request) -> web.Response:
        self.arrivals.append(time.perf_counter())
        body = await request.json()
        self.bodies.append(body)
        self.headers_log.append(dict(request.headers))
        return web.json_response({
            'id': 'chatcmpl-1',
            'object': 'chat.completion',
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': 'ok'},
                'finish_reason': 'stop',
            }],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 1, 'total_tokens': 6},
        })

    async def start(self) -> int:
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, 'localhost', 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()


def _make_trace_file(records, tmp_path):
    path = os.path.join(tmp_path, 'trace.jsonl')
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    return path


def _make_args(trace_path, port, **kwargs):
    return Arguments(
        url=f'http://localhost:{port}/v1/chat/completions',
        dataset='workload_trace',
        dataset_path=trace_path,
        open_loop=True,
        rate=1,
        **kwargs,
    )


async def _run_e2e(records, tmp_path, dataset_args=None, warmup_num=0):
    """Spin up echo server, run full pipeline, return (server, results)."""
    server = EchoServer()
    await server.start()
    try:
        path = _make_trace_file(records, tmp_path)
        extra = {}
        if dataset_args:
            extra['dataset_args'] = dataset_args
        if warmup_num:
            extra['warmup_num'] = warmup_num
        args = _make_args(path, server.port, **extra)
        for attr in ('parallel', 'number', 'rate'):
            v = getattr(args, attr)
            if isinstance(v, list):
                setattr(args, attr, v[0])

        api_plugin = OpenaiPlugin(args)
        dataset_plugin = WorkloadTraceDatasetPlugin(args)

        requests = []
        for messages in dataset_plugin.build_messages():
            request = api_plugin.build_request(messages)
            requests.append(request)

        async def request_gen():
            for r in requests:
                yield r, False

        queue = asyncio.Queue()
        results: list[BenchmarkData] = []

        client = AioHttpClient(args, api_plugin)
        try:
            strategy = OpenLoopStrategy(args, api_plugin, client, queue, request_gen())
            await strategy.run()

            while not queue.empty():
                results.append(await queue.get())
        finally:
            await client.__aexit__(None, None, None)

        return server, results
    except Exception:
        await server.stop()
        raise


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


class TestE2EArrivalTiming:

    def test_arrival_intervals_match_trace(self, tmp_path):
        """Requests arrive at intervals matching the trace timestamps (±80ms)."""
        records = [
            {'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'a'}]}, 'timestamp': 100.0},
            {'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'b'}]}, 'timestamp': 100.3},
            {'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'c'}]}, 'timestamp': 100.8},
        ]

        async def go():
            server, _ = await _run_e2e(records, tmp_path)
            await server.stop()
            return server

        server = _run(go())
        assert len(server.arrivals) == 3
        intervals = [server.arrivals[i + 1] - server.arrivals[i] for i in range(2)]
        assert abs(intervals[0] - 0.3) < 0.08, f'interval[0]={intervals[0]:.3f}, expected ~0.3'
        assert abs(intervals[1] - 0.5) < 0.08, f'interval[1]={intervals[1]:.3f}, expected ~0.5'

    def test_speed_scaling_compresses_intervals(self, tmp_path):
        """speed=2.0 halves the inter-arrival gaps."""
        records = [
            {'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'a'}]}, 'timestamp': 0.0},
            {'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'b'}]}, 'timestamp': 0.6},
        ]

        async def go():
            server, _ = await _run_e2e(records, tmp_path, dataset_args={'speed': 2.0})
            await server.stop()
            return server

        server = _run(go())
        assert len(server.arrivals) == 2
        interval = server.arrivals[1] - server.arrivals[0]
        assert abs(interval - 0.3) < 0.08, f'interval={interval:.3f}, expected ~0.3'


class TestE2EBodyPassthrough:

    def test_body_arrives_verbatim(self, tmp_path):
        """All body fields reach the server unchanged; no body meta keys leak."""
        body = {
            'model': 'qwen-72b',
            'messages': [{'role': 'user', 'content': 'hello'}],
            'temperature': 0.9,
            'max_tokens': 100,
            'top_p': 0.95,
            'seed': 42,
        }
        records = [{'body': body, 'timestamp': 0.0}]

        async def go():
            server, _ = await _run_e2e(records, tmp_path)
            await server.stop()
            return server

        server = _run(go())
        received = server.bodies[0]
        assert received['temperature'] == 0.9
        assert received['max_tokens'] == 100
        assert received['top_p'] == 0.95
        assert received['seed'] == 42
        assert received['model'] == 'qwen-72b'
        for key in received:
            assert not key.startswith(BODY_META_PREFIX), f'body meta key leaked: {key}'

    def test_body_meta_headers_applied(self, tmp_path):
        """Per-request headers from trace are merged into HTTP request headers."""
        records = [{
            'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}]},
            'timestamp': 0.0,
            'headers': {'X-Canary': 'true'},
        }]

        async def go():
            server, _ = await _run_e2e(records, tmp_path)
            await server.stop()
            return server

        server = _run(go())
        assert server.headers_log[0].get('X-Canary') == 'true'

    def test_hop_by_hop_headers_stripped(self, tmp_path):
        """Hop-by-hop headers from trace data are stripped before sending."""
        records = [{
            'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}]},
            'timestamp': 0.0,
            'headers': {'X-Custom': 'keep', 'Host': 'evil.com', 'Connection': 'close'},
        }]

        async def go():
            server, _ = await _run_e2e(records, tmp_path)
            await server.stop()
            return server

        server = _run(go())
        sent_headers = server.headers_log[0]
        assert sent_headers.get('X-Custom') == 'keep'
        # Host should be set by aiohttp to the actual target, not the trace value
        assert sent_headers.get('Host') != 'evil.com'

    def test_request_id_propagated(self, tmp_path):
        """request_id flows through to BenchmarkData.request_id (not trace_id)."""
        records = [{
            'body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}]},
            'timestamp': 0.0,
            'request_id': 'req-abc-123',
        }]

        async def go():
            server, results = await _run_e2e(records, tmp_path)
            await server.stop()
            return results

        results = _run(go())
        assert len(results) == 1
        assert results[0].request_id == 'req-abc-123'
        assert results[0].trace_id is None


class TestE2EWarmup:

    def test_warmup_excluded_from_results(self, tmp_path):
        """Warmup requests have is_warmup=True; non-warmup have is_warmup=False."""
        records = [
            {'body': {'model': 'm', 'messages': [{'role': 'user', 'content': str(i)}]}, 'timestamp': float(i) * 0.1}
            for i in range(5)
        ]

        async def go():
            server, results = await _run_e2e(records, tmp_path, warmup_num=2)
            await server.stop()
            return results

        results = _run(go())
        assert len(results) == 5
        warmup_flags = [r.is_warmup for r in results]
        assert sum(warmup_flags) == 2

    def test_warmup_continuous_timing(self, tmp_path):
        """Warmup and benchmark requests share a single continuous arrival schedule."""
        records = [
            {'body': {'model': 'm', 'messages': [{'role': 'user', 'content': str(i)}]}, 'timestamp': float(i) * 0.2}
            for i in range(4)
        ]

        async def go():
            server, _ = await _run_e2e(records, tmp_path, warmup_num=2)
            await server.stop()
            return server

        server = _run(go())
        assert len(server.arrivals) == 4
        intervals = [server.arrivals[i + 1] - server.arrivals[i] for i in range(3)]
        for idx, interval in enumerate(intervals):
            assert abs(interval - 0.2) < 0.08, f'interval[{idx}]={interval:.3f}, expected ~0.2'
