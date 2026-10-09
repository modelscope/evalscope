"""A request with unavailable usage must not discard the rest of a perf run."""

import asyncio
import json
import sqlite3
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import aiohttp
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.core.pipeline import run_benchmark_pipeline
from evalscope.perf.plugin.api.openai_responses_api import OpenAIResponsesPlugin
from evalscope.perf.utils.benchmark_util import BenchmarkData, MetricsAccumulator


@pytest.fixture
def provider() -> Iterator[str]:
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            mode = request['model']
            response = {
                'id': 'resp_fixture',
                'object': 'response',
                'status': 'completed',
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'ok'}]}],
            }
            if mode == 'valid':
                response['usage'] = {'input_tokens': 7, 'output_tokens': 3}
            if mode == 'sse-missing':
                event = {'type': 'response.output_text.delta', 'delta': 'ok', 'sequence_number': 0}
                body = f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'.encode()
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
        yield f'http://127.0.0.1:{server.server_port}/v1/responses'
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize('mode', ['sse-missing', 'json-missing'])
@pytest.mark.parametrize('fallback', [False, True])
@pytest.mark.parametrize('next_model', ['valid', 'json-missing'])
def test_missing_usage_does_not_abort_pipeline(
    provider: str, tmp_path: Path, mode: str, fallback: bool, next_model: str, caplog: pytest.LogCaptureFixture
) -> None:
    args = Arguments(model='test-model', api='openai_responses', log_every_n_query=100)
    args.number = 2
    args.parallel = 1
    args.outputs_dir = str(tmp_path)
    plugin = OpenAIResponsesPlugin(args)
    if fallback:

        class Tokenizer:
            def encode(self, text: str, **kwargs: object) -> list[int]:
                return list(range(len(text)))

        plugin.tokenizer = Tokenizer()

    records: list[BenchmarkData] = []

    async def go() -> tuple:
        queue: asyncio.Queue[BenchmarkData] = asyncio.Queue()
        async with aiohttp.ClientSession() as session:
            for model in [mode, next_model]:
                data = await plugin.process_request(session, provider, {}, {'model': model, 'input': 'answer'})
                assert data.success is True
                records.append(data)

        async def producer() -> None:
            for data in records:
                await queue.put(data)

        result = await run_benchmark_pipeline(producer(), queue, args, plugin)
        await asyncio.wait_for(queue.join(), timeout=1)
        return result

    metrics, _, _, db_path = asyncio.run(go())
    expected_successes = 2 if fallback else int(next_model == 'valid')
    assert metrics.total_requests == 2
    assert metrics.succeed_requests == expected_successes
    assert metrics.failed_requests == 2 - expected_successes
    assert records[0].success is fallback
    assert records[0].response_messages
    assert records[0].generated_text == 'ok'
    assert records[0].completed_time >= records[0].start_time > 0
    if not fallback:
        assert 'Unable to finalize request metrics' in records[0].error
        assert 'Unable to retrieve usage information' in caplog.text

    with sqlite3.connect(db_path) as con:
        rows = con.execute('SELECT success, prompt_tokens, completion_tokens FROM result ORDER BY rowid').fetchall()
    assert len(rows) == 2
    if not fallback:
        assert rows[0] == (0, None, None)
        if next_model == 'valid':
            assert rows[1] == (1, 7, 3)
            assert metrics.avg_prompt_tokens == 7
            assert metrics.avg_completion_tokens == 3
        else:
            assert rows[1] == (0, None, None)
            assert metrics.avg_prompt_tokens == -1
            assert metrics.avg_completion_tokens == -1
    else:
        assert rows[0] == (1, 6, 2)
        assert rows[1] == ((1, 7, 3) if next_model == 'valid' else (1, 6, 2))


def test_unrelated_parser_errors_still_propagate() -> None:
    class InvalidPlugin:
        def parse_responses(self, responses: list, **kwargs: object) -> tuple[int, int]:
            raise ValueError('Invalid tokenizer configuration')

    with pytest.raises(ValueError, match='Invalid tokenizer configuration'):
        MetricsAccumulator().update(BenchmarkData(success=True), InvalidPlugin())
