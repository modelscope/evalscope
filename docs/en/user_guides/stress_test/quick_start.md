# Quick Start
Below is a quick guide for using EvalScope to conduct model inference performance testing. It supports OpenAI API format model services and various dataset formats, making it convenient for users to perform performance evaluations.

## Environment Preparation

EvalScope supports usage in Python environments. Users can install EvalScope via pip or from source. Here are examples of both installation methods:

::::{tab-set}
:::{tab-item} pip installation
```shell
# Install additional dependencies
pip install evalscope[perf] -U
```
:::

:::{tab-item} Source installation
```shell
git clone https://github.com/modelscope/evalscope.git
cd evalscope
pip install -e '.[perf]'
```
:::
::::

## Basic Usage
You can start the model inference performance testing tool using the following two methods (command line/Python script):

The example below demonstrates performance testing of the Qwen2.5-0.5B-Instruct model using the vLLM framework on an A100, with fixed input of 1024 tokens and output of 1024 tokens. Users can modify parameters according to their needs.

::::{tab-set}
:::{tab-item} Command line startup
```bash
evalscope perf \
  --parallel 1 10 50 100 200 \
  --number 10 20 100 200 400 \
  --model Qwen2.5-0.5B-Instruct \
  --url http://127.0.0.1:8801/v1/chat/completions \
  --api openai \
  --dataset random \
  --max-tokens 1024 \
  --min-tokens 1024 \
  --prefix-length 0 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --extra-args '{"ignore_eos": true}'
```
:::

:::{tab-item} Python script startup
```python
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments

task_cfg = Arguments(
    parallel=[1, 10, 50, 100, 200],
    number=[10, 20, 100, 200, 400],
    model='Qwen2.5-0.5B-Instruct',
    url='http://127.0.0.1:8801/v1/chat/completions',
    api='openai',
    dataset='random',
    min_tokens=1024,
    max_tokens=1024,
    prefix_length=0,
    min_prompt_length=1024,
    max_prompt_length=1024,
    tokenizer_path='Qwen/Qwen2.5-0.5B-Instruct',
    extra_args={'ignore_eos': True}
)
results = run_perf_benchmark(task_cfg)
```
:::
::::

**Parameter description**:

- `parallel`: Number of concurrent requests, multiple values can be passed, separated by spaces
- `number`: Number of total requests for each concurrency, multiple values can be passed, separated by spaces (corresponding one-to-one with `parallel`)
- `url`: Request URL address
- `model`: Name of the model used
- `api`: API service used, default is `openai`
- `dataset`: Dataset name, here it's `random`, indicating randomly generated dataset, for specific usage instructions [refer to](./examples.md#using-the-random-dataset); for more available (multimodal) datasets, please refer to [Dataset Configuration](./parameters.md#dataset-configuration)
- `tokenizer-path`: Model's tokenizer path, used to calculate the number of tokens (necessary for random datasets)
- `extra-args`: Additional parameters in the request, passed as a JSON format string, e.g., `{"ignore_eos": true}` indicates ignoring the end token

```{seealso}
[Complete parameter description](./parameters.md)
```

### Output Results

The output test report summary is shown in the image below, including basic information, metrics for each concurrency, and performance test suggestions:

![multi_perf](./images/multi_perf.png)

```{note}
- The stress test report in the diagram is a summary of test results across multiple concurrency levels, allowing users to compare model performance under different concurrency settings. No summary report is generated for a single concurrency level.
- This report will be saved in `outputs/<timestamp>/<model>/performance_summary.txt`, which users can view as needed.
- For explanations of the metrics in the table below, please refer to the "Metric Descriptions" section that follows. Results will be saved in `outputs/<timestamp>/<model>/benchmark.log`.
```

Additionally, the test results for each concurrency level are output separately, including metrics such as the number of requests, successful requests, failed requests, average latency, and average latency per token for each concurrency level.

```text
Benchmarking summary:
+-----------------------------------+------------+
| Key                               |      Value |
+===================================+============+
| Time taken for tests (s)          |    62.9998 |
+-----------------------------------+------------+
| Number of concurrency             |   200      |
+-----------------------------------+------------+
| Total requests                    |   400      |
+-----------------------------------+------------+
| Succeed requests                  |   400      |
+-----------------------------------+------------+
| Failed requests                   |     0      |
+-----------------------------------+------------+
| Output token throughput (tok/s)   |  6501.61   |
+-----------------------------------+------------+
| Total token throughput (tok/s)    | 13379.1    |
+-----------------------------------+------------+
| Request throughput (req/s)        |     6.3492 |
+-----------------------------------+------------+
| Average latency (s)               |    30.9964 |
+-----------------------------------+------------+
| Average time to first token (s)   |     1.3071 |
+-----------------------------------+------------+
| Average time per output token (s) |     0.029  |
+-----------------------------------+------------+
| Average input tokens per request  |  1083.2    |
+-----------------------------------+------------+
| Average output tokens per request |  1024      |
+-----------------------------------+------------+
2025-05-16 11:36:33,122 - evalscope - INFO - 
Percentile results:
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
| Percentiles | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output (tok/s) | Total (tok/s) |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
|     10%     |  0.3371  |   0.0   |  0.0275  |   28.8971   |     1029     |     1024      |    30.5721     |    62.9394    |
|     25%     |  0.4225  | 0.0218  |  0.0281  |   29.7338   |     1061     |     1024      |    32.1195     |    65.5693    |
|     50%     |  0.9366  | 0.0263  |  0.0291  |   30.8739   |     1073     |     1024      |    33.1733     |    67.9478    |
|     66%     |  1.5892  | 0.0293  |  0.0296  |   31.5315   |     1080     |     1024      |    33.9593     |    69.8569    |
|     75%     |  2.0769  | 0.0308  |  0.0298  |   31.8956   |     1084     |     1024      |    34.4398     |    70.9794    |
|     80%     |  2.3656  | 0.0319  |  0.0299  |   32.3317   |     1088     |     1024      |    34.7858     |    71.7338    |
|     90%     |  3.1024  |  0.039  |  0.0304  |   33.4968   |     1197     |     1024      |    35.4505     |    73.0745    |
|     95%     |  3.5413  | 0.0572  |  0.0308  |   34.1158   |     1251     |     1024      |    35.8045     |    74.3043    |
|     98%     |  3.8131  | 0.1462  |  0.0309  |   34.5008   |     1292     |     1024      |    36.0191     |    77.1365    |
|     99%     |  3.8955  | 0.1761  |  0.031   |   34.5951   |     1343     |     1024      |    36.1281     |    79.4895    |
+-------------+----------+---------+----------+-------------+--------------+---------------+----------------+---------------+
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
| Average inter-token latency | Average time interval between generating each output token | Total inter-token latency / Successful requests |
| Average input tokens per request | Average number of input tokens per request | Total input tokens / Successful requests |
| Average output tokens per request | Average number of output tokens per request | Total output tokens / Successful requests |

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

### Using WandB

First, install wandb and obtain the corresponding [API Key](https://wandb.ai/settings):
```bash
pip install wandb
```

To upload the test results to the wandb server and visualize them, add the following parameters when launching the evaluation:
```bash
# ...
--visualizer wandb
--name 'name_of_wandb_log'
```

For example:

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)


### Using SwanLab

First, install SwanLab and obtain the corresponding [API Key](https://swanlab.cn/space/~/settings):
```bash
pip install swanlab
```

To upload the test results to the swanlab server and visualize them, add the following parameters when launching the evaluation:
```bash
# ...
--visualizer swanlab
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

### Using ClearML
Please install ClearML using the following command:
```bash
pip install clearml
```

Initialize the ClearML server:
```bash
clearml-init
```

Add the following parameters before starting the test:
```bash
# You can use the CLEARML_PROJECT_NAME environment variable to specify the project name
--visualizer clearml
--name 'name_of_clearml_task'
```

![clearml sample](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/clearml_vis.jpg)

