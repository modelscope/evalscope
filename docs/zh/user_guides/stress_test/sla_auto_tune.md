# SLA 自动调优

在 `PerfConfig` 中设置 `SLAConfig` 即可搜索最大并发或请求到达率。SLA 与普通压测复用同一个异步运行引擎，并返回类型化 `SLAResult`。

```python
from evalscope.perf import SLAConfig

sla = SLAConfig(
    variable='concurrency',
    criteria=[{'p99_latency': '<=2.0', 'success_rate': '>=99'}],
    lower_bound=1,
    upper_bound=64,
    repetitions=3,
    pass_ratio=1.0,
)
```

同一字典内为 AND，多字典之间为 OR。默认每次重复运行都必须通过。约束搜索使用区间扩张和二分，复测边界相邻点；观测到非单调结果时改为区间扫描。

吞吐优化使用 `objective='max_rps'` 或 `max_output_tps`，`min_latency` 用于最小时延。算法先指数粗扫，再扫描峰值邻域；相同时选择更低负载。
