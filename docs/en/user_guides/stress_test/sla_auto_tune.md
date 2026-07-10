# SLA Auto-tuning

Attach `SLAConfig` to `PerfConfig` to search concurrency or request rate. SLA tuning uses the same asynchronous run engine as normal suites and returns a typed `SLAResult`.

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

Criteria within a mapping use AND semantics; multiple mappings use OR semantics. Every repeated run must pass by default. Constraint search brackets and then bisects the boundary, verifies the neighboring load, and falls back to a scan if observed results are non-monotonic.

For throughput optimization, use `objective='max_rps'` or `objective='max_output_tps'`; `min_latency` minimizes average latency. Optimization uses an exponential coarse scan followed by a local scan and chooses the lower load on ties.
