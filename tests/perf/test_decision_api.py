"""Decision perf must measure complete, validated requests without generation defaults."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from evalscope.api.model.decision import DecisionInput
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.api.decision_api import AutoTrustPerfPlugin, SystemOnePerfPlugin
from evalscope.perf.utils.perf_constants import Metrics


def request(kind: str) -> DecisionInput:
    return DecisionInput(kind=kind, state={'text': '今日天气晴朗。'}, instructions='Decide.',
                         criteria={'A': 'First', 'B': 'Second'} if kind == 'choice' else
                         [str(i) for i in range(6)] if kind == 'score' else None)


def response(protocol: str, kind: str) -> dict:
    q = request(kind)
    if protocol == 'autotrust':
        options = list(q.criteria.values()) if kind == 'choice' else ['false', 'true'] if kind == 'noul' else list(map(str, range(6)))
        return {'model': 'v1', 'kind': kind, 'options': options, 'choice': options[0], 'choice_index': 0,
                'probabilities': [1.0] + [0.0] * (len(options) - 1),
                'usage': {'prompt_tokens': 12, 'completion_tokens': 1}}
    a = {'type': kind}
    if kind == 'noul':
        a['noul'] = 0.0
    else:
        a['probabilities'] = {label: float(i == 0) for i, label in enumerate(q.labels)}
        if kind == 'choice':
            a['choice'] = 'A'
        else:
            a.update(score=0.0, legend=dict(zip(q.labels, q.criteria)))
    return {'model': 'v1', 'answers': {'answer': a}, 'usage': {'input_tokens': 12, 'output_tokens': 0}}


@pytest.mark.parametrize('protocol,cls', [('systemone', SystemOnePerfPlugin), ('autotrust', AutoTrustPerfPlugin)])
@pytest.mark.parametrize('kind', ['choice', 'noul', 'score'])
def test_decision_http_validation(protocol: str, cls: type, kind: str) -> None:
    async def go() -> None:
        payload = response(protocol, kind)
        calls = []
        async def handle(req: web.Request) -> web.Response:
            calls.append(await req.json())
            return web.json_response(payload)
        app = web.Application()
        app.router.add_post('/v1/' + ('systemone' if protocol == 'systemone' else 'decide'), handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        args = Arguments(model='v1', api=protocol, url=f'http://127.0.0.1:{port}/v1', stream=False)
        plugin = cls(args)
        body = request(kind).systemone_payload('v1') if protocol == 'systemone' else request(kind).autotrust_payload()
        assert plugin.build_request(body) == body
        try:
            async with aiohttp.ClientSession() as client:
                good = await plugin.process_request(client, args.url, {}, body)
                assert good.success
                assert good.query_latency > 0
                assert good.prompt_tokens == 12
                assert good.first_chunk_latency == good.time_per_output_token == 0
                assert not any(key in calls[0] for key in ('stream', 'temperature', 'max_tokens'))
                payload.clear()
                invalid = await plugin.process_request(client, args.url, {}, body)
                assert not invalid.success
                assert 'Invalid decision response' in invalid.error
                assert len(calls) == 2
        finally:
            await runner.cleanup()
    asyncio.run(go())


@pytest.mark.parametrize('api,cls', [('systemone', SystemOnePerfPlugin), ('autotrust', AutoTrustPerfPlugin)])
def test_reject_generation_controls(api: str, cls: type) -> None:
    plugin = cls(Arguments(model='v1', api=api))
    body = request('choice').systemone_payload('v1') if api == 'systemone' else request('choice').autotrust_payload()
    with pytest.raises(ValueError, match='generation controls'):
        plugin.build_request({**body, 'max_tokens': 1})


def test_concurrency_sweep_and_input_metrics(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/v1/systemone'
            assert 'max_tokens' not in body
            raw = json.dumps(response('systemone', 'choice')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / 'requests.jsonl'
    path.write_text(json.dumps(request('choice').systemone_payload('v1')) + '\n')
    try:
        results = run_perf_benchmark(Arguments(
            model='v1', api='systemone', url=f'http://127.0.0.1:{server.server_port}/v1',
            dataset='line_by_line', dataset_path=str(path), stream=False, parallel=[1, 4], number=[8, 8],
            warmup_num=2, sleep_interval=0, outputs_dir=str(tmp_path / 'perf'),
        ))
        assert len(results) == 2
        for result in results.values():
            metrics = result['metrics'].to_dict()
            assert metrics[Metrics.SUCCEED_REQUESTS] == 8
            assert metrics[Metrics.FAILED_REQUESTS] == 0
            assert metrics[Metrics.INPUT_TOKEN_THROUGHPUT] > 0
            assert metrics.get(Metrics.AVERAGE_TIME_TO_FIRST_TOKEN, 0) == 0
            assert result['percentiles'].rows[0].latency is not None
            assert result['percentiles'].rows[0].ttft is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
