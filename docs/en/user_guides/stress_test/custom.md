# Extending Perf

Custom behavior is split into orthogonal workload and protocol adapters.

```python
from evalscope.perf.domain.workload import SingleTurnItem, WorkloadMeta
from evalscope.perf.workloads import WorkloadSource, register_workload

@register_workload
class MyWorkload(WorkloadSource):
    meta = WorkloadMeta(name='my_workload', mode='single_turn', protocols=frozenset({'openai_chat'}))

    async def iter_items(self, run):
        yield SingleTurnItem(messages='hello')
```

A workload declares its mode, supported data sources, tokenizer requirement, parallel-generation capability, and compatible protocols. Duplicate names fail immediately.

Custom wire formats subclass `ProtocolAdapter`, declare `ProtocolMeta`, and register with `register_protocol`. Protocol adapters only encode requests and consume `TransportEvent`; HTTP, SSE framing, timeouts, and status handling remain in `HttpTransport`.

For data-only customization, prefer the built-in `custom`, `custom_multi_turn`, or `line_by_line` workloads.
