# 快速开始

安装性能压测依赖：

```bash
pip install 'evalscope[perf]'
```

对 OpenAI 兼容服务运行闭环压测：

```bash
evalscope perf \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --workload prompt --prompt '解释 KV Cache。' \
  --mode closed_loop --concurrency 8 --requests 100
```

Python API 使用不可变的类型化配置：

```python
from evalscope.perf import BenchmarkSuite, ClosedLoopLoad, PerfConfig, TargetConfig, WorkloadConfig, run_perf

result = run_perf(
    PerfConfig(
        target=TargetConfig(model='Qwen/Qwen2.5-7B-Instruct', base_url='http://127.0.0.1:8000/v1'),
        workload=WorkloadConfig(name='prompt', prompt='解释 KV Cache。'),
        suite=BenchmarkSuite(loads=[ClosedLoopLoad(concurrency=8, request_count=100)]),
    )
)
```

`run_perf` 返回 `PerfSuiteResult`；异步程序使用 `await async_run_perf(config)`。每个 suite 会生成 manifest、类型化 JSON 汇总、每个负载点的 SQLite 原始观测和 HTML 报告。

输出位于 `<output-root>/<run-id>/`。run id 已存在时默认失败，只有显式设置 `--overwrite` 才覆盖。
