from __future__ import annotations

import asyncio
import json
import os
from typing import Type

from evalscope.perf.config.models import ClosedLoopLoad, ConversationLoad, OpenLoopLoad, PerfConfig
from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.domain.errors import PerfError, PerfRunError
from evalscope.perf.domain.result import ArtifactManifest, RunResult
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.engine.context import RunContext
from evalscope.perf.engine.observers import ProgressObserver, VisualizerObserver
from evalscope.perf.engine.schedulers import ClosedLoopScheduler, ConversationScheduler, OpenLoopScheduler
from evalscope.perf.metrics.definitions import METRICS
from evalscope.perf.protocols import create_protocol
from evalscope.perf.results import SQLiteResultStore, summarize_store
from evalscope.perf.serving import ManagedTarget
from evalscope.perf.transport import AioHttpTransport
from evalscope.perf.workloads import WorkloadContext, workload_registry


class RunEngine:
    """Shared lifecycle for every performance benchmark mode."""

    def __init__(self, config: PerfConfig, run_id: str, spec: ResolvedRunSpec, suite_dir: str) -> None:
        self.config = config
        self.run_id = run_id
        self.spec = spec
        self.suite_dir = suite_dir
        self.run_dir = os.path.join(suite_dir, 'runs', spec.load_id)

    async def run(self) -> RunResult:
        os.makedirs(self.run_dir, exist_ok=False)
        db_path = os.path.join(self.run_dir, 'observations.sqlite')
        store = SQLiteResultStore(db_path, self.config.runtime.db_commit_interval)
        protocol = create_protocol(self.config.target)
        workload_class = workload_registry.get(self.config.workload.name)
        workload = workload_class(WorkloadContext(self.config))

        connector_limit = self._connector_limit()
        transport = AioHttpTransport(self.config.target, connector_limit=connector_limit)
        queue = asyncio.Queue(maxsize=self.config.runtime.queue_size)
        context = RunContext(
            run_id=self.run_id,
            config=self.config,
            spec=self.spec,
            transport=transport,
            protocol=protocol,
            store=store,
            queue=queue,
        )
        if self.config.runtime.progress:
            context.observers.append(ProgressObserver(os.path.join(self.suite_dir, 'progress.json')))
        if self.config.runtime.visualizer:
            context.observers.append(
                VisualizerObserver(
                    self.config.runtime.visualizer,
                    self.config.runtime.visualizer_project,
                    self.config.runtime.visualizer_name or self.run_id,
                )
            )
        target_log = os.path.join(self.run_dir, 'target.log')
        managed_target = ManagedTarget(self.config.target, target_log)

        try:
            store.open()
            await workload.prepare()
            async with managed_target, transport:
                await self._connection_test(context, managed_target)
                items = workload.iter_items(self.spec)
                scheduler = self._scheduler(context, items)
                await self._run_scheduler(context, scheduler)

            summary, percentiles, trace_summary, workload_summary = summarize_store(
                store,
                self.config.metrics.last_window_seconds,
                self.config.metrics.steady_state_warmup_ratio,
            )
            result = RunResult(
                run_id=self.run_id,
                run_spec=self.spec,
                summary=summary,
                percentiles=percentiles,
                trace_summary=trace_summary,
                workload_summary=workload_summary,
                metric_definitions={name: METRICS[name]
                                    for name in percentiles
                                    if name in METRICS},
                artifacts=ArtifactManifest(
                    root=self.run_dir,
                    files={
                        'config': os.path.join(self.run_dir, 'run_config.json'),
                        'summary': os.path.join(self.run_dir, 'summary.json'),
                        'percentiles': os.path.join(self.run_dir, 'percentiles.json'),
                        'observations': db_path,
                        'trace_summary': os.path.join(self.run_dir, 'trace_summary.json') if trace_summary else None,
                        'workload_timeline': os.path.join(self.run_dir, 'workload_timeline.json')
                        if workload_summary else None,
                    },
                ),
            )
            self._write_result_files(result)
            return result
        except PerfRunError as e:
            if e.run_id == self.run_id:
                raise
            raise PerfRunError(self.run_id, e.stage, str(e)) from e
        except PerfError as e:
            raise PerfRunError(self.run_id, 'execution', str(e)) from e
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise PerfRunError(self.run_id, 'execution', str(e)) from e
        finally:
            context.close_observers()
            store.close()

    async def _run_scheduler(self, context: RunContext, scheduler) -> None:
        consumer_task = asyncio.create_task(context.consume())
        scheduler_task = asyncio.create_task(scheduler.run())
        try:
            done, _ = await asyncio.wait({scheduler_task, consumer_task}, return_when=asyncio.FIRST_COMPLETED)
            if consumer_task in done:
                await consumer_task
                raise PerfRunError(self.run_id, 'consumer', 'consumer stopped unexpectedly')
            await scheduler_task
        except BaseException:
            context.cancelled.set()
            if not scheduler_task.done():
                scheduler_task.cancel()
                await asyncio.gather(scheduler_task, return_exceptions=True)
            if not consumer_task.done():
                await context.finish()
                try:
                    await consumer_task
                except BaseException as consumer_error:
                    raise PerfRunError(self.run_id, 'consumer', str(consumer_error)) from consumer_error
            raise
        else:
            await context.finish()
            await consumer_task
        finally:
            for task in (scheduler_task, consumer_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(scheduler_task, consumer_task, return_exceptions=True)

    def _connector_limit(self) -> int:
        load = self.spec.load
        if isinstance(load, OpenLoopLoad):
            return load.max_outstanding
        return load.concurrency

    def _scheduler(self, context: RunContext, items):
        load = self.spec.load
        if isinstance(load, ClosedLoopLoad):
            return ClosedLoopScheduler(context, items)
        if isinstance(load, OpenLoopLoad):
            return OpenLoopScheduler(context, items)
        if isinstance(load, ConversationLoad):
            return ConversationScheduler(context, items)
        raise TypeError(f'Unsupported load type: {type(load).__name__}')

    async def _connection_test(self, context: RunContext, target: ManagedTarget) -> None:
        if self.config.target.skip_connection_test:
            return
        deadline = context.clock() + min(float(self.config.target.total_timeout or 60), 60.0)
        if self.config.target.protocol == 'openai_rerank':
            item = SingleTurnItem(messages={'query': 'hello', 'documents': ['hello']})
        else:
            item = SingleTurnItem(messages='hello')
        while True:
            target.ensure_running()
            observation = await context.execute(item, is_warmup=True)
            if observation.success:
                return
            if observation.status_code is not None and 400 <= observation.status_code < 500:
                raise PerfRunError(self.run_id, 'connection_test', observation.error or 'request was rejected')
            if context.clock() >= deadline:
                raise PerfRunError(self.run_id, 'connection_test', observation.error or 'connection timed out')
            await asyncio.sleep(1)

    def _write_result_files(self, result: RunResult) -> None:
        files = result.artifacts.files
        payloads = {
            files['config']: self.spec.model_dump(mode='json'),
            files['summary']: result.summary.model_dump(mode='json'),
            files['percentiles']: {
                name: value.model_dump(mode='json')
                for name, value in result.percentiles.items()
            },
        }
        if result.trace_summary and files['trace_summary']:
            payloads[files['trace_summary']] = result.trace_summary.model_dump(mode='json')
        if result.workload_summary and files['workload_timeline']:
            payloads[files['workload_timeline']] = result.workload_summary.model_dump(mode='json')
        for path, payload in payloads.items():
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
