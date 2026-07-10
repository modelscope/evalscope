# 扩展 Perf

自定义能力分成相互独立的 workload 与 protocol adapter。

```python
from evalscope.perf.domain.workload import SingleTurnItem, WorkloadMeta
from evalscope.perf.workloads import WorkloadSource, register_workload

@register_workload
class MyWorkload(WorkloadSource):
    meta = WorkloadMeta(name='my_workload', mode='single_turn', protocols=frozenset({'openai_chat'}))

    async def iter_items(self, run):
        yield SingleTurnItem(messages='hello')
```

Workload 必须声明模式、数据源能力、tokenizer 需求、并行生成能力和兼容协议；重复名称会立即报错。

自定义 wire 格式继承 `ProtocolAdapter`、声明 `ProtocolMeta`，并用 `register_protocol` 注册。Protocol 只编码请求和消费 `TransportEvent`；HTTP、SSE framing、超时和状态码由 `HttpTransport` 统一处理。

仅自定义数据时，优先使用内置 `custom`、`custom_multi_turn` 或 `line_by_line` workload。
