import asyncio
import json
from aiohttp import web

from evalscope.perf import (
    BenchmarkSuite,
    ClosedLoopLoad,
    ConversationLoad,
    OpenLoopLoad,
    OutputConfig,
    PerfConfig,
    TargetConfig,
    WorkloadConfig,
    async_run_perf,
)
from evalscope.perf.reporting import load_suite


async def _handler(request: web.Request) -> web.StreamResponse:
    await request.json()
    response = web.StreamResponse(status=200, headers={'Content-Type': 'text/event-stream'})
    await response.prepare(request)
    chunks = [
        {
            'choices': [{
                'delta': {
                    'content': 'ok'
                }
            }]
        },
        {
            'choices': [],
            'usage': {
                'prompt_tokens': 2,
                'completion_tokens': 1
            }
        },
    ]
    for chunk in chunks:
        await response.write(f'data: {json.dumps(chunk)}\n\n'.encode())
    await response.write(b'data: [DONE]\n\n')
    await response.write_eof()
    return response


async def _run(tmp_path, load, *, run_id=None, delay=0):
    app = web.Application()

    async def handler(request):
        if delay:
            await asyncio.sleep(delay)
        return await _handler(request)

    app.router.add_post('/v1/chat/completions', handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    dataset = tmp_path / 'prompts.txt'
    dataset.write_text('hello\nworld\n', encoding='utf-8')
    config = PerfConfig(
        target=TargetConfig(model='fake', base_url=f'http://127.0.0.1:{port}/v1'),
        workload=WorkloadConfig(name='custom', path=str(dataset), data_source='local'),
        suite=BenchmarkSuite(loads=[load]),
        output=OutputConfig(root=str(tmp_path), run_id=run_id or f'run-{load.mode}'),
    )
    try:
        result = await async_run_perf(config)
    finally:
        await runner.cleanup()
    return result


def test_closed_loop_typed_result_and_artifacts(tmp_path) -> None:
    result = asyncio.run(_run(tmp_path, ClosedLoopLoad(concurrency=2, request_count=4)))
    assert result.runs[0].summary.succeeded == 4
    assert result.runs[0].summary.total == 4
    loaded = load_suite(result.artifacts.root)
    assert loaded.run_id == result.run_id
    loaded_from_manifest = load_suite(result.artifacts.files['manifest'])
    assert loaded_from_manifest.run_id == result.run_id


def test_open_loop_is_bounded_and_observable(tmp_path) -> None:
    result = asyncio.run(_run(tmp_path, OpenLoopLoad(request_rate=100, request_count=10, max_outstanding=2)))
    summary = result.runs[0].summary
    assert summary.total == 10
    assert summary.succeeded + summary.failed + summary.dropped == 10


def test_open_loop_records_overflow_without_backpressure(tmp_path) -> None:
    result = asyncio.run(
        _run(
            tmp_path,
            OpenLoopLoad(
                request_rate=1000,
                request_count=20,
                max_outstanding=1,
                arrival='constant',
            ),
            run_id='overflow',
            delay=0.05,
        )
    )
    assert result.runs[0].summary.dropped > 0
    assert result.runs[0].summary.total == 20


def test_concurrent_suites_do_not_share_runtime_state(tmp_path) -> None:

    async def run_both():
        return await asyncio.gather(
            _run(tmp_path, ClosedLoopLoad(concurrency=1, request_count=3), run_id='concurrent-a'),
            _run(tmp_path, ClosedLoopLoad(concurrency=2, request_count=4), run_id='concurrent-b'),
        )

    first, second = asyncio.run(run_both())
    assert first.runs[0].summary.total == 3
    assert second.runs[0].summary.total == 4


def test_conversation_scheduler_accumulates_context(tmp_path) -> None:

    async def run_conversation():
        received = []

        async def handler(request: web.Request):
            payload = await request.json()
            received.append(payload['messages'])
            return web.json_response({
                'choices': [{
                    'message': {
                        'content': 'assistant reply'
                    }
                }],
                'usage': {
                    'prompt_tokens': len(payload['messages']),
                    'completion_tokens': 2
                },
            })

        app = web.Application()
        app.router.add_post('/v1/chat/completions', handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        dataset = tmp_path / 'conversations.jsonl'
        dataset.write_text(
            json.dumps([
                {
                    'role': 'user',
                    'content': 'first'
                },
                {
                    'role': 'assistant',
                    'content': 'reference'
                },
                {
                    'role': 'user',
                    'content': 'second'
                },
            ]) + '\n',
            encoding='utf-8',
        )
        config = PerfConfig(
            target=TargetConfig(model='fake', base_url=f'http://127.0.0.1:{port}/v1'),
            workload=WorkloadConfig(name='custom_multi_turn', path=str(dataset), data_source='local'),
            suite=BenchmarkSuite(loads=[ConversationLoad(concurrency=1, conversation_count=1)]),
            output=OutputConfig(root=str(tmp_path), run_id='conversation'),
        )
        try:
            result = await async_run_perf(config)
        finally:
            await runner.cleanup()
        return result, received

    result, received = asyncio.run(run_conversation())
    assert result.runs[0].summary.total == 2
    assert len(received[-1]) == 3
    assert received[-1][1] == {'role': 'assistant', 'content': 'assistant reply'}
