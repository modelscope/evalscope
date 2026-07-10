# Quick Start

Install the performance dependencies:

```bash
pip install 'evalscope[perf]'
```

Run a closed-loop benchmark against an OpenAI-compatible service:

```bash
evalscope perf \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --workload prompt --prompt 'Explain KV caching.' \
  --mode closed_loop --concurrency 8 --requests 100
```

The Python API uses immutable, typed configuration:

```python
from evalscope.perf import (
    BenchmarkSuite,
    ClosedLoopLoad,
    PerfConfig,
    TargetConfig,
    WorkloadConfig,
    run_perf,
)

result = run_perf(
    PerfConfig(
        target=TargetConfig(model='Qwen/Qwen2.5-7B-Instruct', base_url='http://127.0.0.1:8000/v1'),
        workload=WorkloadConfig(name='prompt', prompt='Explain KV caching.'),
        suite=BenchmarkSuite(loads=[ClosedLoopLoad(concurrency=8, request_count=100)]),
    )
)
```

`run_perf` returns `PerfSuiteResult`. Async applications use `await async_run_perf(config)`.
Each suite writes a manifest, typed JSON summaries, one SQLite observation store per load point, and an HTML report.

The output root is `<output-root>/<run-id>/`; existing run IDs fail unless `--overwrite` is explicit.
