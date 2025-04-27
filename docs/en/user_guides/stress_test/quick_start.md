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
| Time taken for tests (s)          | 4.2364                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Number of concurrency             | 5                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Total requests                    | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Succeed requests                  | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Failed requests                   | 0                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Output token throughput (tok/s)   | 880.6965                                                        |
+-----------------------------------+-----------------------------------------------------------------+
| Total token throughput (tok/s)    | 1117.9251                                                       |
+-----------------------------------+-----------------------------------------------------------------+
| Request throughput (req/s)        | 4.721                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average latency (s)               | 0.9463                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average time to first token (s)   | 0.214                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average time per output token (s) | 0.0038                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average input tokens per request  | 50.25                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average output tokens per request | 186.55                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average package latency (s)       | 0.0039                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average package per request       | 186.35                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Expected number of requests       | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Result DB path                    | outputs/20250424_113806/Qwen2.5-0.5B-Instruct/benchmark_data.db |
+-----------------------------------+-----------------------------------------------------------------+
2025-04-24 11:38:12,015 - evalscope - INFO - 
Percentile results:
+------------+----------+---------+----------+-------------+--------------+---------------+--------------------------+-------------------------+
| Percentile | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output throughput(tok/s) | Total throughput(tok/s) |
+------------+----------+---------+----------+-------------+--------------+---------------+--------------------------+-------------------------+
|    10%     |  0.0168  |   0.0   |  0.001   |   0.3334    |      42      |      61       |         155.5959         |        198.9264         |
|    25%     |  0.0172  | 0.0042  |  0.0044  |   0.7491    |      47      |      87       |         210.4683         |         253.708         |
|    50%     |  0.0174  | 0.0045  |  0.0045  |    0.999    |      49      |      204      |         215.9736         |        265.5097         |
|    66%     |  0.0269  | 0.0046  |  0.0045  |   1.1166    |      52      |      242      |         216.7906         |        281.8135         |
|    75%     |  0.0329  | 0.0046  |  0.0045  |   1.2857    |      55      |      279      |         217.8717         |        321.8351         |
|    80%     |  0.9821  | 0.0046  |  0.0045  |   1.3452    |      58      |      287      |         219.2701         |        334.9922         |
|    90%     |  0.9952  | 0.0047  |  0.0046  |   1.5973    |      62      |      348      |         220.0688         |        381.5638         |
|    95%     |  0.9995  | 0.0048  |  0.0062  |   1.6652    |      66      |      361      |         220.1942         |        417.4862         |
|    98%     |  0.9995  | 0.0052  |  0.0062  |   1.6652    |      66      |      361      |         220.1942         |        417.4862         |
|    99%     |  0.9995  | 0.0076  |  0.0062  |   1.6652    |      66      |      361      |         220.1942         |        417.4862         |
+------------+----------+---------+----------+-------------+--------------+---------------+--------------------------+-------------------------+
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
| Total token throughput (tok/s) | Average number of tokens processed per second | Total tokens / Time taken for tests |
| Request throughput (req/s) | Average number of successful queries processed per second | Successful requests / Time taken for tests |
| Total latency | Sum of latency times for all successful requests | Sum of all successful request latencies |
| Average latency | Average time from sending a request to receiving a complete response | Total latency / Successful requests |
| Average time to first token | Average time from sending a request to receiving the first response token | Total first chunk latency / Successful requests |
| Average time per output token | Average time required to generate each output token (exclude first token) | Total time per output token / Successful requests |
| Average input tokens per request | Average number of input tokens per request | Total input tokens / Successful requests |
| Average output tokens per request | Average number of output tokens per request | Total output tokens / Successful requests |
| Average package latency | Average delay time for receiving each data package | Total package time / Total packages |
| Average package per request | Average number of data packages received per request | Total packages / Successful requests |

**Percentile Metrics**

| Metric | Explanation |
|--------|-------------|
| Time to First Token | The time from sending a request to generating the first token (in seconds), assessing the initial packet delay. |
| Inter-token Latency | The time interval between generating each output token (in seconds), assessing the smoothness of output. |
| Time per Output Token | The time required to generate each output token (excluding the first token, in seconds), assessing decoding speed. |
| Latency | The time from sending a request to receiving a complete response (in seconds): Time to First Token + Time per Output Token * Output tokens. |
| Input Tokens | The number of tokens input in the request. |
| Output Tokens | The number of tokens generated in the response. |
| Output Throughput | The number of tokens output per second: Output tokens / Latency. |
| Total Throughput | The number of tokens processed per second: (Input tokens + Output tokens) / Latency. |


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
