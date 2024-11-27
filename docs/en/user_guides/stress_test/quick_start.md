# Quick Start

## Environment Preparation
::::{tab-set}
:::{tab-item} Install via pip
```shell
# Install additional dependencies
pip install evalscope[perf] -U
```
:::

:::{tab-item} Install from source
```shell
git clone https://github.com/modelscope/evalscope.git
cd evalscope
pip install -e '.[perf]'
```
:::
::::

## Basic Usage
You can start the model inference performance stress testing tool using the following two methods:

::::{tab-set}
:::{tab-item} Command Line Start
```bash
evalscope perf \
    --url "http://127.0.0.1:8000/v1/chat/completions" \
    --parallel 1 \
    --model qwen2.5 \
    --number 15 \
    --api openai \
    --dataset openqa \
    --stream
```
:::

:::{tab-item} Start with Python Script
```python
from evalscope.perf.main import run_perf_benchmark

task_cfg = {"url": "http://127.0.0.1:8000/v1/chat/completions",
            "parallel": 1,
            "model": "qwen2.5",
            "number": 15,
            "api": "openai",
            "dataset": "openqa",
            "stream": True}
run_perf_benchmark(task_cfg)
```
:::
::::
Parameter Descriptions:

- `url`: The URL for the request
- `parallel`: The number of tasks for concurrent requests
- `model`: The name of the model used
- `number`: The number of requests
- `api`: The API service used
- `dataset`: The name of the dataset
- `stream`: Whether to enable streaming

### Output Results
```text
Benchmarking summary: 
+----------------------------------------------+------------------------------------------------+
| key                                          | Value                                          |
+==============================================+================================================+
| Time taken for tests (seconds)              | 7.539                                          |
+----------------------------------------------+------------------------------------------------+
| Number of concurrency                        | 1                                              |
+----------------------------------------------+------------------------------------------------+
| Total requests                               | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Succeeded requests                           | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Failed requests                              | 0                                              |
+----------------------------------------------+------------------------------------------------+
| Average QPS                                  | 1.99                                           |
+----------------------------------------------+------------------------------------------------+
| Average latency                              | 0.492                                          |
+----------------------------------------------+------------------------------------------------+
| Average time to first token                  | 0.026                                          |
+----------------------------------------------+------------------------------------------------+
| Throughput (average output tokens per second) | 334.006                                        |
+----------------------------------------------+------------------------------------------------+
| Average time per output token                | 0.00299                                        |
+----------------------------------------------+------------------------------------------------+
| Average package per request                  | 167.867                                        |
+----------------------------------------------+------------------------------------------------+
| Average package latency                      | 0.003                                          |
+----------------------------------------------+------------------------------------------------+
| Average input tokens per request             | 40.133                                         |
+----------------------------------------------+------------------------------------------------+
| Average output tokens per request            | 167.867                                        |
+----------------------------------------------+------------------------------------------------+
| Expected number of requests                  | 15                                             |
+----------------------------------------------+------------------------------------------------+
| Result DB path                               | ./outputs/qwen2.5_benchmark_20241107_201413.db |
+----------------------------------------------+------------------------------------------------+

Percentile results: 
+------------+---------------------+---------+
| Percentile | First Chunk Latency | Latency |
+------------+---------------------+---------+
|    10%     |       0.0178        | 0.1577  |
|    25%     |       0.0183        | 0.2358  |
|    50%     |       0.0199        | 0.4311  |
|    66%     |       0.0218        | 0.6317  |
|    75%     |       0.0429        | 0.7121  |
|    80%     |       0.0432        | 0.7957  |
|    90%     |       0.0432        | 0.9153  |
|    95%     |       0.0433        | 0.9897  |
|    98%     |       0.0433        | 0.9897  |
|    99%     |       0.0433        | 0.9897  |
+------------+---------------------+---------+
```

### Metric Descriptions

| **Metric**                                  | **Description**                  | **Value**      |
|------------------------------------------|---------------------------------|-----------------|
| Total requests                          | Total number of requests        | 15              |
| Succeeded requests                      | Number of successful requests   | 15              |
| Failed requests                         | Number of failed requests       | 0               |
| Average QPS                            | Average requests per second     | 1.99            |
| Average latency                         | Average latency for all requests| 0.492           |
| Throughput (average output tokens per second) | Average output tokens per second| 334.006        |
| Average time to first token            | Average delay for the first token| 0.026          |
| Average input tokens per request        | Average number of input tokens per request | 40.133  |
| Average output tokens per request       | Average number of output tokens per request | 167.867 |
| Average time per output token           | Average time for each output token | 0.00299      |
| Average package per request             | Average number of packages per request | 167.867 |
| Average package latency                  | Average latency for each package| 0.003           |
| Percentile of time to first token (p10, ..., p99) | Percentiles for the first token latency |       |
| Percentile of request latency (p10, ..., p99) | Percentiles for request latency |                |