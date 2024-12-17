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
+-----------------------------------+-----------------------------------------------------+
| Key                               | Value                                               |
+===================================+=====================================================+
| Time taken for tests (s)          | 10.739                                              |
+-----------------------------------+-----------------------------------------------------+
| Number of concurrency             | 1                                                   |
+-----------------------------------+-----------------------------------------------------+
| Total requests                    | 15                                                  |
+-----------------------------------+-----------------------------------------------------+
| Succeed requests                  | 15                                                  |
+-----------------------------------+-----------------------------------------------------+
| Failed requests                   | 0                                                   |
+-----------------------------------+-----------------------------------------------------+
| Throughput(average tokens/s)      | 324.059                                             |
+-----------------------------------+-----------------------------------------------------+
| Average QPS                       | 1.397                                               |
+-----------------------------------+-----------------------------------------------------+
| Average latency (s)               | 0.696                                               |
+-----------------------------------+-----------------------------------------------------+
| Average time to first token (s)   | 0.029                                               |
+-----------------------------------+-----------------------------------------------------+
| Average time per output token (s) | 0.00309                                             |
+-----------------------------------+-----------------------------------------------------+
| Average input tokens per request  | 50.133                                              |
+-----------------------------------+-----------------------------------------------------+
| Average output tokens per request | 232.0                                               |
+-----------------------------------+-----------------------------------------------------+
| Average package latency (s)       | 0.003                                               |
+-----------------------------------+-----------------------------------------------------+
| Average package per request       | 232.0                                               |
+-----------------------------------+-----------------------------------------------------+
| Expected number of requests       | 15                                                  |
+-----------------------------------+-----------------------------------------------------+
| Result DB path                    | ./outputs/20241216_194204/qwen2.5/benchmark_data.db |
+-----------------------------------+-----------------------------------------------------+

Percentile results: 
+------------+----------+----------+-------------+--------------+---------------+----------------------+
| Percentile | TTFT (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Throughput(tokens/s) |
+------------+----------+----------+-------------+--------------+---------------+----------------------+
|    10%     |  0.0202  |  0.0027  |   0.1846    |      41      |      50       |       270.8324       |
|    25%     |  0.0209  |  0.0028  |   0.2861    |      44      |      83       |       290.0714       |
|    50%     |  0.0233  |  0.0028  |   0.7293    |      49      |      250      |       335.644        |
|    66%     |  0.0267  |  0.0029  |   0.9052    |      50      |      308      |       340.2603       |
|    75%     |  0.0437  |  0.0029  |   0.9683    |      53      |      325      |       341.947        |
|    80%     |  0.0438  |  0.003   |   1.0799    |      58      |      376      |       342.7985       |
|    90%     |  0.0439  |  0.0032  |   1.2474    |      62      |      424      |       345.5268       |
|    95%     |  0.0463  |  0.0033  |   1.3038    |      66      |      431      |       348.1648       |
|    98%     |  0.0463  |  0.0035  |   1.3038    |      66      |      431      |       348.1648       |
|    99%     |  0.0463  |  0.0037  |   1.3038    |      66      |      431      |       348.1648       |
+------------+----------+----------+-------------+--------------+---------------+----------------------+
```

### Metric Descriptions

| Metric                                | Description                                                                                 |
|--------------------------------------|-------------------------------------------------------------------------------------------|
| Time taken for tests (s)             | Time used for tests (seconds)                                                             |
| Number of concurrency                | Number of concurrent requests                                                             |
| Total requests                       | Total number of requests                                                                  |
| Succeed requests                     | Number of successful requests                                                             |
| Failed requests                      | Number of failed requests                                                                 |
| Throughput (average tokens/s)        | Throughput (average number of tokens processed per second)                                |
| Average QPS                          | Average number of queries per second (Queries Per Second)                                 |
| Average latency (s)                  | Average latency time (seconds)                                                            |
| Average time to first token (s)      | Average time to first token (seconds)                                                     |
| Average time per output token (s)    | Average time per output token (seconds)                                                   |
| Average input tokens per request     | Average number of input tokens per request                                                |
| Average output tokens per request    | Average number of output tokens per request                                               |
| Average package latency (s)          | Average package latency time (seconds)                                                    |
| Average package per request          | Average number of packages per request                                                    |
| Expected number of requests          | Expected number of requests                                                               |
| Result DB path                       | Result database path                                                                      |
| **Percentile**                       | **Data is divided into 100 equal parts, and the nth percentile indicates that n% of the data points are below this value** |
| TTFT (s)                             | Time to First Token, the time to generate the first token                                 |
| TPOT (s)                             | Time Per Output Token, the time to generate each output token                             |
| Latency (s)                          | Latency time, the time between request and response                                       |
| Input tokens                         | Number of input tokens                                                                    |
| Output tokens                        | Number of output tokens                                                                   |
| Throughput (tokens/s)                | Throughput, the number of tokens processed per second                                     |