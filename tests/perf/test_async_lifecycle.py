import asyncio
import pytest
from aiohttp import web
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.benchmark import run_benchmark
from evalscope.perf.core import pipeline
from evalscope.perf.core.http_client import AioHttpClient
from evalscope.perf.core.strategies.closed_loop import ClosedLoopStrategy
from evalscope.perf.core.strategies.multi_turn import MultiTurnStrategy
from evalscope.perf.core.strategies.open_loop import OpenLoopStrategy
from evalscope.perf.utils.db_util import get_result_db_path
from evalscope.perf.utils.handler import exception_handler


def _make_args(**kwargs: Any) -> Arguments:
    args = Arguments(model='test-model', api='openai', number=2, parallel=2, rate=-1, **kwargs)
    args.number = 2
    args.parallel = 2
    args.rate = -1
    return args


def test_benchmark_pipelines_own_independent_completion_events(monkeypatch: pytest.MonkeyPatch) -> None:
    completion_events: List[asyncio.Event] = []

    async def fake_consumer(
        queue: asyncio.Queue,
        args: Arguments,
        api_plugin: Any,
        completed_event: asyncio.Event,
    ) -> Tuple[None, None, None, str]:
        completion_events.append(completed_event)
        while not (completed_event.is_set() and queue.empty()):
            try:
                await asyncio.wait_for(queue.get(), timeout=0.01)
            except asyncio.TimeoutError:
                continue
            queue.task_done()
        return None, None, None, ''

    async def run() -> None:
        async def produce(queue: asyncio.Queue) -> None:
            await queue.put(object())

        first_queue: asyncio.Queue = asyncio.Queue()
        second_queue: asyncio.Queue = asyncio.Queue()
        args = _make_args()
        await asyncio.gather(
            pipeline.run_benchmark_pipeline(produce(first_queue), first_queue, args, None),
            pipeline.run_benchmark_pipeline(produce(second_queue), second_queue, args, None),
        )

    monkeypatch.setattr(pipeline, 'statistic_benchmark_metric', fake_consumer)
    asyncio.run(run())

    assert len(completion_events) == 2
    assert completion_events[0] is not completion_events[1]
    assert all(event.is_set() for event in completion_events)


@pytest.mark.parametrize('failure_source', ['producer', 'consumer'])
def test_benchmark_pipeline_propagates_failure_and_cancels_peer(
    monkeypatch: pytest.MonkeyPatch,
    failure_source: str,
) -> None:
    peer_cancelled = asyncio.Event()

    async def consumer(
        queue: asyncio.Queue,
        args: Arguments,
        api_plugin: Any,
        completed_event: asyncio.Event,
    ) -> Tuple[None, None, None, str]:
        if failure_source == 'consumer':
            raise LookupError('consumer failed')
        try:
            await asyncio.Event().wait()
        finally:
            peer_cancelled.set()
        return None, None, None, ''

    async def producer() -> None:
        if failure_source == 'producer':
            raise ValueError('producer failed')
        try:
            await asyncio.Event().wait()
        finally:
            peer_cancelled.set()

    async def run() -> None:
        current_task = asyncio.current_task()
        expected_error = ValueError if failure_source == 'producer' else LookupError
        with pytest.raises(expected_error):
            await pipeline.run_benchmark_pipeline(producer(), asyncio.Queue(), _make_args(), None)
        assert peer_cancelled.is_set()
        assert all(task is current_task or task.done() for task in asyncio.all_tasks())

    monkeypatch.setattr(pipeline, 'statistic_benchmark_metric', consumer)
    asyncio.run(run())


@pytest.mark.parametrize('strategy_class', [ClosedLoopStrategy, OpenLoopStrategy])
def test_strategy_failure_cancels_in_flight_requests(strategy_class: type) -> None:
    blocked_request_cancelled = asyncio.Event()

    class FakeClient:

        async def post(self, request: Dict[str, int]) -> SimpleNamespace:
            if request['id'] == 1:
                await asyncio.sleep(0)
                raise RuntimeError('request failed')
            try:
                await asyncio.Event().wait()
            finally:
                blocked_request_cancelled.set()
            return SimpleNamespace(is_warmup=False)

    async def request_generator() -> AsyncIterator[Tuple[dict, bool]]:
        yield {'id': 1}, False
        yield {'id': 2}, False

    async def run() -> None:
        strategy = strategy_class(_make_args(), None, FakeClient(), asyncio.Queue(), request_generator())
        with pytest.raises(RuntimeError, match='request failed'):
            await strategy.run()
        assert blocked_request_cancelled.is_set()
        current_task = asyncio.current_task()
        assert all(task is current_task or task.done() for task in asyncio.all_tasks())

    asyncio.run(run())


def test_multi_turn_strategy_failure_cancels_workers() -> None:
    blocked_worker_cancelled = asyncio.Event()

    async def run() -> None:
        strategy = MultiTurnStrategy(_make_args(), None, None, asyncio.Queue(), [[]])

        async def worker(worker_id: int) -> None:
            if worker_id == 0:
                await asyncio.sleep(0)
                raise RuntimeError('worker failed')
            try:
                await asyncio.Event().wait()
            finally:
                blocked_worker_cancelled.set()

        strategy._worker = worker
        with pytest.raises(RuntimeError, match='worker failed'):
            await strategy._run_phase(budget=2, is_warmup=False)
        assert blocked_worker_cancelled.is_set()
        current_task = asyncio.current_task()
        assert all(task is current_task or task.done() for task in asyncio.all_tasks())

    asyncio.run(run())


def test_exception_handler_preserves_original_exception() -> None:

    @exception_handler
    def fail() -> None:
        raise KeyError('original')

    with pytest.raises(KeyError, match='original'):
        fail()

    @exception_handler
    async def async_fail() -> None:
        raise LookupError('async original')

    with pytest.raises(LookupError, match='async original'):
        asyncio.run(async_fail())

    @exception_handler
    async def async_generator_fail() -> AsyncIterator[None]:
        if False:
            yield
        raise RuntimeError('generator original')

    async def consume_generator() -> None:
        async for _ in async_generator_fail():
            pass

    with pytest.raises(RuntimeError, match='generator original'):
        asyncio.run(consume_generator())


def test_existing_result_database_raises_file_exists_error(tmp_path) -> None:
    db_path = tmp_path / 'benchmark_data.db'
    db_path.touch()

    with pytest.raises(FileExistsError, match=str(db_path)):
        get_result_db_path(SimpleNamespace(outputs_dir=str(tmp_path)))


def test_aiohttp_client_context_returns_self_and_closes() -> None:
    async def run() -> None:
        client = AioHttpClient(_make_args(), None)
        async with client as entered:
            assert entered is client
            assert not client.client.closed
        assert client.client.closed
        await client.__aexit__(None, None, None)

    asyncio.run(run())


class LocalOpenAIServer:

    def __init__(self) -> None:
        self.request_count = 0
        self.runner: Any = None

    async def handle(self, request: web.Request) -> web.Response:
        await request.json()
        self.request_count += 1
        return web.json_response({
            'id': 'chatcmpl-local',
            'object': 'chat.completion',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'ok'
                },
                'finish_reason': 'stop',
            }],
            'usage': {
                'prompt_tokens': 1,
                'completion_tokens': 1,
                'total_tokens': 2
            },
        })

    async def start(self) -> int:
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self.handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        return site._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        await self.runner.cleanup()


def test_local_benchmarks_run_sequentially_and_concurrently(tmp_path) -> None:
    def make_args(port: int, name: str) -> Arguments:
        output_dir = tmp_path / name
        output_dir.mkdir()
        args = _make_args(
            url=f'http://127.0.0.1:{port}/v1/chat/completions',
            prompt='hello',
            stream=False,
            no_test_connection=True,
            outputs_dir=str(output_dir),
        )
        args.outputs_dir = str(output_dir)
        return args

    async def run() -> int:
        server = LocalOpenAIServer()
        port = await server.start()
        try:
            await run_benchmark(make_args(port, 'sequential-one'))
            await run_benchmark(make_args(port, 'sequential-two'))
            await asyncio.gather(
                run_benchmark(make_args(port, 'concurrent-one')),
                run_benchmark(make_args(port, 'concurrent-two')),
            )
            return server.request_count
        finally:
            await server.close()

    assert asyncio.run(run()) == 8
