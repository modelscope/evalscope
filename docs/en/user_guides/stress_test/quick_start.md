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


```{important}
To accurately measure the Time to First Token (TTFT) metric, you need to include the `--stream` parameter in your request. Otherwise, TTFT will be the same as Latency.
```

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

**Metrics**

| Metric | Explanation | Formula |
|--------|-------------|---------|
| Time taken for tests | Total time from the start to the end of the test process | Last request end time - First request start time |
| Number of concurrency | Number of clients sending requests simultaneously | Preset value |
| Total requests | Total number of requests sent during the testing process | Successful requests + Failed requests |
| Succeed requests | Number of requests completed successfully and returning expected results | Direct count |
| Failed requests | Number of requests that failed to complete successfully | Direct count |
| Throughput (average tokens/s) | Average number of tokens processed per second | Total output tokens / Time taken for tests |
| Average QPS | Average number of successful queries processed per second | Successful requests / Time taken for tests |
| Total latency | Sum of latency times for all successful requests | Sum of all successful request latencies |
| Average latency | Average time from sending a request to receiving a complete response | Total latency / Successful requests |
| Average time to first token | Average time from sending a request to receiving the first response token | Total first chunk latency / Successful requests |
| Average time per output token | Average time required to generate each output token | Total time per output token / Successful requests |
| Average input tokens per request | Average number of input tokens per request | Total input tokens / Successful requests |
| Average output tokens per request | Average number of output tokens per request | Total output tokens / Successful requests |
| Average package latency | Average delay time for receiving each data package | Total package time / Total packages |
| Average package per request | Average number of data packages received per request | Total packages / Successful requests |

**Percentile Metrics**

| Metric | Explanation |
|--------|-------------|
| Time to First Token | Time from sending a request to generating the first token (in seconds) |
| Time Per Output Token | Time needed to generate each output token (in seconds) |
| Latency | Time from sending a request to receiving a complete response (in seconds) |
| Input tokens | Number of tokens inputted in the request |
| Output tokens | Number of tokens generated in the response |
| Throughput | Number of tokens outputted per second (tokens/s) |


## Visualizing Test Results

First, install wandb and obtain the corresponding [API Key](https://wandb.ai/settings):
```bash
pip install wandb
```

To upload the test results to the wandb server and visualize them, add the following parameters when launching the evaluation:
```bash
# ...
--wandb-api-key 'wandb_api_key'
--name 'name_of_wandb_log'
```

For example:

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)