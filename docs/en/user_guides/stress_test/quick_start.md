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
    --parallel 5 \
    --model qwen2.5 \
    --number 20 \
    --api openai \
    --dataset openqa \
    --stream
```
:::

:::{tab-item} Start with Python Script
```python
from evalscope.perf.main import run_perf_benchmark

task_cfg = {"url": "http://127.0.0.1:8000/v1/chat/completions",
            "parallel": 5,
            "model": "qwen2.5",
            "number": 20,
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
+-----------------------------------+-----------------------------------------------------------------+
| Key                               | Value                                                           |
+===================================+=================================================================+
| Time taken for tests (s)          | 2.9033                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Number of concurrency             | 5                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Total requests                    | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Succeed requests                  | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Failed requests                   | 0                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Output token throughput (tok/s)   | 881.7537                                                        |
+-----------------------------------+-----------------------------------------------------------------+
| Request throughput (req/s)        | 6.8887                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average latency (s)               | 0.7221                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average time to first token (s)   | 0.0264                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average time per output token (s) | 0.0056                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average input tokens per request  | 15.0                                                            |
+-----------------------------------+-----------------------------------------------------------------+
| Average output tokens per request | 128.0                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average package latency (s)       | 0.0055                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average package per request       | 127.0                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Expected number of requests       | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Result DB path                    | outputs/20250409_202434/Qwen2.5-0.5B-Instruct/benchmark_data.db |
+-----------------------------------+-----------------------------------------------------------------+
2025-04-09 20:24:37,412 - evalscope - db_util.py - summary_result - 215 - INFO - 
Percentile results:
+------------+----------+---------+-------------+--------------+---------------+----------------------+
| Percentile | TTFT (s) | ITL (s) | Latency (s) | Input tokens | Output tokens | Throughput(tokens/s) |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
|    10%     |  0.0139  |  0.005  |   0.6796    |      15      |      128      |       165.8655       |
|    25%     |  0.0151  | 0.0052  |   0.6883    |      15      |      128      |       170.323        |
|    50%     |  0.0164  | 0.0054  |   0.7431    |      15      |      128      |       184.9038       |
|    66%     |  0.0192  | 0.0056  |   0.7487    |      15      |      128      |       185.0271       |
|    75%     |  0.0197  | 0.0058  |   0.7644    |      15      |      128      |       188.1981       |
|    80%     |  0.0663  | 0.0059  |   0.7707    |      15      |      128      |       188.2957       |
|    90%     |  0.0677  | 0.0062  |   0.7726    |      15      |      128      |       188.7613       |
|    95%     |  0.0715  | 0.0065  |   0.7727    |      15      |      128      |       189.5965       |
|    98%     |  0.0715  | 0.0067  |   0.7727    |      15      |      128      |       189.5965       |
|    99%     |  0.0715  | 0.0073  |   0.7727    |      15      |      128      |       189.5965       |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
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
| Output token throughput (tok/s) | Average number of tokens processed per second | Total output tokens / Time taken for tests |
| Request throughput (req/s) | Average number of successful queries processed per second | Successful requests / Time taken for tests |
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
| Inter-token Latency | Time needed to generate each output token (in seconds) |
| Latency | Time from sending a request to receiving a complete response (in seconds) |
| Input tokens | Number of tokens inputted in the request |
| Output tokens | Number of tokens generated in the response |
| Throughput | Number of tokens outputted per second (tokens/s) |


## Visualizing Test Results

### Visualizing with WandB

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


### Visualizing Test Results with SwanLab

First, install SwanLab and obtain the corresponding [API Key](https://swanlab.cn/space/~/settings):
```bash
pip install swanlab
```

To upload the test results to the swanlab server and visualize them, add the following parameters when launching the evaluation:
```bash
# ...
--swanlab-api-key 'swanlab_api_key'
--name 'name_of_swanlab_log'
```

For example:

![swanlab sample](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/swanlab.png)

If you prefer to use [SwanLab in local dashboard mode](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html), install swanlab dashboard first:

```bash
pip install 'swanlab[dashboard]'
```

and set the following parameters instead:
```bash
--swanlab-api-key local
```

Then, use `swanlab watch <log_path>` to launch the local visualization dashboard.
