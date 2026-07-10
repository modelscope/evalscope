from __future__ import annotations

import asyncio
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from evalscope.perf.config.models import PerfConfig
from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.domain.observation import RequestObservation
from evalscope.perf.domain.workload import SingleTurnItem
from evalscope.perf.metrics.request import RequestMetricsAggregator
from evalscope.perf.protocols.base import ProtocolAdapter
from evalscope.perf.results.store import ResultStore
from evalscope.perf.transport.base import HttpTransport
from evalscope.utils.logger import get_logger

_END = object()
logger = get_logger()


@dataclass
class RunContext:
    """State owned exclusively by one benchmark run."""

    run_id: str
    config: PerfConfig
    spec: ResolvedRunSpec
    transport: HttpTransport
    protocol: ProtocolAdapter
    store: ResultStore
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    clock: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    rng: Optional[random.Random] = None
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    aggregator: RequestMetricsAggregator = field(default_factory=RequestMetricsAggregator)
    observers: List = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random(self.spec.seed)
        self._request_ids = itertools.count()

    def next_request_id(self) -> str:
        return f'{self.spec.load_id}-{next(self._request_ids)}'

    def sample_max_tokens(self, override: Optional[int] = None) -> int:
        if override is not None:
            return override
        value = self.config.generation.max_tokens
        if isinstance(value, tuple):
            return self.rng.randint(value[0], value[1])
        return value

    async def execute(
        self,
        item: SingleTurnItem,
        *,
        is_warmup: bool,
        scheduled_time: Optional[float] = None,
        trace_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        is_first_turn: bool = False,
        is_last_turn: bool = False,
        max_tokens: Optional[int] = None,
        outstanding: Optional[int] = None,
    ) -> RequestObservation:
        request_id = self.next_request_id()
        dispatch_time = self.clock()
        try:
            request = self.protocol.build_request(
                item,
                self.config.generation,
                max_tokens=self.sample_max_tokens(max_tokens),
            )
            state = self.protocol.new_state(request)
            async for event in self.transport.events(request):
                self.protocol.consume_event(state, event)
            state = self.protocol.finalize(state)
            return RequestObservation(
                run_id=self.run_id,
                request_id=request_id,
                trace_id=trace_id,
                turn_index=turn_index,
                is_first_turn=is_first_turn,
                is_last_turn=is_last_turn,
                is_warmup=is_warmup,
                scheduled_time=scheduled_time,
                dispatch_time=dispatch_time,
                start_time=state.start_time,
                first_token_time=state.first_token_time,
                completed_time=state.completed_time,
                success=state.success,
                status_code=state.status_code,
                error_type=state.error_type,
                error=state.error,
                outstanding=outstanding,
                prompt_tokens=state.prompt_tokens,
                completion_tokens=state.completion_tokens,
                cached_tokens=state.cached_tokens,
                accepted_draft_tokens=state.accepted_draft_tokens,
                proposed_draft_tokens=state.proposed_draft_tokens,
                prompt_token_source=state.prompt_token_source,
                completion_token_source=state.completion_token_source,
                cache_token_source=state.cache_token_source,
                chunk_times=state.chunk_times,
                generated_text=state.generated_text,
                request_payload=request.body,
                response_payloads=state.response_payloads,
                metadata=item.metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            now = self.clock()
            return RequestObservation(
                run_id=self.run_id,
                request_id=request_id,
                trace_id=trace_id,
                turn_index=turn_index,
                is_first_turn=is_first_turn,
                is_last_turn=is_last_turn,
                is_warmup=is_warmup,
                scheduled_time=scheduled_time,
                dispatch_time=dispatch_time,
                start_time=dispatch_time,
                completed_time=now,
                error_type=type(e).__name__,
                error=str(e),
                outstanding=outstanding,
            )

    async def emit(self, observation: RequestObservation) -> None:
        await self.queue.put(observation)

    async def consume(self) -> None:
        while True:
            value = await self.queue.get()
            try:
                if value is _END:
                    return
                observation: RequestObservation = value
                self.store.write(observation)
                self.aggregator.feed(observation)
                for observer in list(self.observers):
                    try:
                        observer.observe(observation)
                    except Exception as e:
                        self.observers.remove(observer)
                        logger.warning(f'Disabling failed perf observer {type(observer).__name__}: {e}')
                        try:
                            observer.close()
                        except Exception as close_error:
                            logger.warning(f'Failed to close perf observer {type(observer).__name__}: {close_error}')
            finally:
                self.queue.task_done()

    async def finish(self) -> None:
        await self.queue.put(_END)

    def close_observers(self) -> None:
        for observer in self.observers:
            try:
                observer.close()
            except Exception as e:
                logger.warning(f'Failed to close perf observer {type(observer).__name__}: {e}')
