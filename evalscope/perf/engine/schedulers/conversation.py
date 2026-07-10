from __future__ import annotations

import asyncio
from typing import Optional

from evalscope.perf.config.models import ConversationLoad
from evalscope.perf.domain.workload import ConversationItem, SingleTurnItem
from evalscope.perf.engine.schedulers.base import Scheduler


class ConversationScheduler(Scheduler):

    async def run(self) -> None:
        load = self.context.spec.load
        assert isinstance(load, ConversationLoad)
        if self.context.spec.warmup_count:
            await self._phase(self.context.spec.warmup_count, True, None, load)
        deadline = self.context.clock() + load.duration if load.duration is not None else None
        await self._phase(load.conversation_count, False, deadline, load)

    async def _phase(
        self,
        limit: Optional[int],
        is_warmup: bool,
        deadline: Optional[float],
        load: ConversationLoad,
    ) -> None:
        claimed = 0
        claim_lock = asyncio.Lock()

        async def worker(worker_id: int) -> None:
            nonlocal claimed
            while not self.context.cancelled.is_set():
                async with claim_lock:
                    if deadline is not None and self.context.clock() >= deadline:
                        return
                    if limit is not None and claimed >= limit:
                        return
                    trace_number = claimed
                    claimed += 1
                item = await self.cursor.next()
                if item is None:
                    return
                if not isinstance(item, ConversationItem):
                    raise TypeError('Conversation scheduler requires conversation work items')
                await self._conversation(item, f'{self.context.spec.load_id}-trace-{trace_number}', is_warmup, load)

        tasks = [asyncio.create_task(worker(index)) for index in range(load.concurrency)]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _conversation(
        self,
        item: ConversationItem,
        trace_id: str,
        is_warmup: bool,
        load: ConversationLoad,
    ) -> None:
        context = []
        turns = item.turns[:load.max_turns] if load.max_turns else item.turns
        for index, turn in enumerate(turns):
            if turn.tool_call_latency:
                await self.context.sleep(turn.tool_call_latency)
            context.extend(message.copy() for message in turn.messages)
            observation = await self.context.execute(
                SingleTurnItem(messages=list(context), metadata=item.metadata),
                is_warmup=is_warmup,
                trace_id=trace_id,
                turn_index=index,
                is_first_turn=index == 0,
                is_last_turn=index == len(turns) - 1,
                max_tokens=turn.max_tokens,
            )
            if not observation.success:
                observation.is_last_turn = True
            await self.context.emit(observation)
            if not observation.success:
                return
            context.append({'role': 'assistant', 'content': observation.generated_text})
