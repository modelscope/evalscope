"""Empty SSE responses must fail without contaminating successful-request metrics."""

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator


def _event(payload: dict) -> bytes:
    return f'data: {json.dumps(payload)}\n\n'.encode()


def _choice(delta: dict) -> bytes:
    return _event({'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta}]})


_USAGE = _event({'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 2}})
_INVALID = {
    'empty': b'',
    'done_only': b'data: [DONE]\n\n',
    'heartbeat_only': b': ping\n\nevent: ping\nid: 1\n\n',
    'truncated': b'data: {"choices":',
    'malformed': b'data: not-json\n\n',
    'unrelated_json': b'data: {"ping": true}\n\n',
    'empty_choices': b'data: {"choices": [], "usage": null}\n\n',
}
_VALID = {
    'text': _choice({'content': 'ok'}) + _USAGE + b'data: [DONE]\n\n',
    'no_done': _choice({'content': 'ok'}) + _USAGE,
    'custom_done': _choice({'content': 'ok'}) + _USAGE + b'data: [FINISHED]\n\n',
    'tools': _choice({'tool_calls': [{'index': 0, 'function': {'name': 'f', 'arguments': '{}'}}]}) + _USAGE,
    'reasoning': _choice({'reasoning_content': 'think'}) + _USAGE,
    'empty_completion': _event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}) + _USAGE,
    'zero_usage': _event({'choices': [], 'usage': {'prompt_tokens': 7, 'completion_tokens': 0}}),
    'completion': _event({'object': 'text_completion', 'choices': [{'index': 0, 'text': 'ok'}]}) + _USAGE,
}
_BODIES = {**_INVALID, **_VALID, 'missing_usage': _choice({'content': 'ok'})}


@pytest.fixture(scope='module')
def provider() -> Iterator[str]:
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            payload = _BODIES[request['model']]
            # Delay headers too: Content-Length: 0 completes an empty response immediately.
            time.sleep(0.01)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 128

    server = Server(('127.0.0.1', 0), Provider)
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


@pytest.mark.parametrize('model', _INVALID)
def test_stream_without_response_events_is_failure(provider: str, model: str) -> None:
    plugin = OpenaiPlugin(Arguments(model=model))
    output = _request(plugin, provider, model)

    assert output.success is False
    assert output.is_stream is True
    assert output.error and 'SSE' in output.error
    assert output.completed_time > output.start_time > 0
    assert output.query_latency == pytest.approx(output.completed_time - output.start_time)
    assert output.first_chunk_latency == 0
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (0, 1)
    assert accumulator.total_latency == 0
    assert accumulator.total_completion_tokens == 0
    assert accumulator.wall_time == pytest.approx(output.query_latency)


@pytest.mark.parametrize('model', _VALID)
def test_valid_streams_remain_successful(provider: str, model: str) -> None:
    plugin = OpenaiPlugin(Arguments(model=model))
    output = _request(plugin, provider, model)
    assert output.success is True
    assert output.error is None
    accumulator = MetricsAccumulator()
    accumulator.update(output, plugin)
    result = accumulator.to_result()
    assert (result.succeed_requests, result.failed_requests) == (1, 0)
    assert accumulator.total_prompt_tokens == 7
    assert accumulator.total_completion_tokens == (0 if model == 'zero_usage' else 2)


def test_valid_response_without_usage_still_requires_tokenizer(provider: str) -> None:
    plugin = OpenaiPlugin(Arguments(model='missing_usage'))
    output = _request(plugin, provider, 'missing_usage')
    assert output.success is True
    with pytest.raises(ValueError, match='tokenizer'):
        MetricsAccumulator().update(output, plugin)


@pytest.mark.parametrize(('parallel', 'all_empty'), [(1, False), (8, False), (32, False), (8, True)])
def test_streams_full_benchmark(provider: str, tmp_path: Path, parallel: int, all_empty: bool) -> None:
    # A shared plugin handles overlapping successful/empty responses through the actual dispatcher and consumer.
    models = ['empty'] * 48 if all_empty else ['empty', 'text', 'truncated', 'tools', 'done_only', 'zero_usage'] * 8
    dataset = tmp_path / 'requests.jsonl'
    dataset.write_text(''.join(json.dumps({'model': model, 'messages': []}) + '\n' for model in models))
    args = Arguments(
        model='mixed', url=provider, api='openai', dataset='line_by_line', dataset_path=str(dataset),
        parallel=parallel, number=48, outputs_dir=str(tmp_path / 'outputs'), warmup_num=0, no_test_connection=True,
    )
    results = run_perf_benchmark(args)
    metrics = results[f'parallel_{parallel}_number_48']['metrics']
    expected_counts = (0, 48) if all_empty else (24, 24)
    assert (metrics.succeed_requests, metrics.failed_requests) == expected_counts
    if not all_empty:
        assert metrics.avg_input_tokens == 7
        assert metrics.avg_output_tokens == pytest.approx(32 / 24, abs=0.001)
    db_paths = list(tmp_path.rglob('*.db'))
    assert len(db_paths) == 1
    with sqlite3.connect(db_paths[0]) as connection:
        rows = connection.execute('SELECT request, success, start_time, completed_time FROM result').fetchall()
    assert len(rows) == 48
    for request, success, start, end in rows:
        assert bool(success) == (json.loads(request)['model'] in _VALID)
        assert end > start > 0
    assert list(tmp_path.rglob('*.html'))
    summaries = list(tmp_path.rglob('benchmark_summary.json'))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert (summary['Success Requests'], summary['Failed Requests']) == expected_counts
