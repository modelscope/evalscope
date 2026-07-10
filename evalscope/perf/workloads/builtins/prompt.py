from typing import AsyncIterator

from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.domain.errors import PerfConfigError
from evalscope.perf.domain.workload import SingleTurnItem, WorkItem, WorkloadMeta
from evalscope.perf.workloads.base import WorkloadSource
from evalscope.perf.workloads.registry import register_workload


@register_workload
class PromptWorkload(WorkloadSource):
    meta = WorkloadMeta(
        name='prompt',
        mode='single_turn',
        requires_dataset=False,
        protocols=frozenset({'openai_chat', 'openai_completions', 'openai_responses', 'openai_embedding'}),
    )

    async def prepare(self) -> None:
        if not self.context.config.workload.prompt:
            raise PerfConfigError('The prompt workload requires workload.prompt')

    async def iter_items(self, run: ResolvedRunSpec) -> AsyncIterator[WorkItem]:
        prompt = self.context.config.workload.prompt
        while True:
            yield SingleTurnItem(messages=prompt)
