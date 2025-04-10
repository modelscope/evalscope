# 快速开始

## 环境准备
::::{tab-set}
:::{tab-item} pip安装
```shell
# 安装额外依赖
pip install evalscope[perf] -U
```
:::

:::{tab-item} 源码安装
```shell
git clone https://github.com/modelscope/evalscope.git
cd evalscope
pip install -e '.[perf]'
```
:::
::::

## 基本使用
可以使用以下两种方式启动模型推理性能压测工具：

::::{tab-set}
:::{tab-item} 命令行启动
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

:::{tab-item} Python脚本启动
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
参数说明：

- `url`: 请求的URL地址
- `parallel`: 并行请求的任务数量
- `model`: 使用的模型名称
- `number`: 请求数量
- `api`: 使用的API服务
- `dataset`: 数据集名称
- `stream`: 是否启用流式处理

```{important}
要准确统计Time to First Token (TTFT)指标，需要在请求中包含`--stream`参数，否则TTFT将与Latency相同。
```

### 输出结果
```text
Benchmarking summary:
+-----------------------------------+-----------------------------------------------------------------+
| Key                               | Value                                                           |
+===================================+=================================================================+
| Time taken for tests (s)          | 3.882                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Number of concurrency             | 5                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Total requests                    | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Succeed requests                  | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Failed requests                   | 0                                                               |
+-----------------------------------+-----------------------------------------------------------------+
| Output token throughput (tok/s)   | 983.7757                                                        |
+-----------------------------------+-----------------------------------------------------------------+
| Total token throughput (tok/s)    | 1242.6641                                                       |
+-----------------------------------+-----------------------------------------------------------------+
| Request throughput (req/s)        | 5.152                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average latency (s)               | 0.8416                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average time to first token (s)   | 0.0247                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average time per output token (s) | 0.0044                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average input tokens per request  | 50.25                                                           |
+-----------------------------------+-----------------------------------------------------------------+
| Average output tokens per request | 190.95                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average package latency (s)       | 0.0043                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Average package per request       | 190.95                                                          |
+-----------------------------------+-----------------------------------------------------------------+
| Expected number of requests       | 20                                                              |
+-----------------------------------+-----------------------------------------------------------------+
| Result DB path                    | outputs/20250410_133556/Qwen2.5-0.5B-Instruct/benchmark_data.db |
+-----------------------------------+-----------------------------------------------------------------+

Percentile results:
+------------+----------+---------+-------------+--------------+---------------+----------------------+
| Percentile | TTFT (s) | ITL (s) | Latency (s) | Input tokens | Output tokens | Throughput(tokens/s) |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
|    10%     |  0.0118  |  0.004  |   0.3434    |      42      |      80       |       215.1057       |
|    25%     |  0.0122  | 0.0041  |   0.5027    |      47      |      113      |       223.166        |
|    50%     |  0.0157  | 0.0042  |   0.7857    |      49      |      194      |       226.2531       |
|    66%     |  0.0161  | 0.0043  |   1.0707    |      52      |      246      |       229.2028       |
|    75%     |  0.0293  | 0.0044  |   1.1826    |      55      |      268      |       229.7484       |
|    80%     |  0.0633  | 0.0044  |   1.2737    |      58      |      290      |       230.8304       |
|    90%     |  0.0654  | 0.0046  |   1.4551    |      62      |      328      |       232.939        |
|    95%     |  0.0655  | 0.0049  |   1.4913    |      66      |      335      |       250.1984       |
|    98%     |  0.0655  | 0.0065  |   1.4913    |      66      |      335      |       250.1984       |
|    99%     |  0.0655  | 0.0072  |   1.4913    |      66      |      335      |       250.1984       |
+------------+----------+---------+-------------+--------------+---------------+----------------------+
```

### 指标说明
| 指标 | 英文名称 | 解释 | 公式 |
|------|----------|------|------|
| 测试总时长 | Time taken for tests | 整个测试过程从开始到结束所花费的总时间 | 最后一个请求结束时间 - 第一个请求开始时间 |
| 并发数 | Number of concurrency | 同时发送请求的客户端数量 | 预设值 |
| 总请求数 | Total requests | 在整个测试过程中发送的所有请求的数量 | 成功请求数 + 失败请求数 |
| 成功请求数 | Succeed requests | 成功完成并返回预期结果的请求数量 | 直接统计 |
| 失败请求数 | Failed requests | 由于各种原因未能成功完成的请求数量 | 直接统计 |
| 输出吞吐量 | Output token throughput | 每秒钟处理的平均标记（token）数 | 总输出token数 / 测试总时长 |
| 总吞吐量 | Total token throughput | 每秒钟处理的平均标记（token）数 | 总输入token数 + 总输出token数 / 测试总时长 |
| 请求吞吐量 | Request throughput | 每秒钟成功处理的平均请求数 | 成功请求数 / 测试总时长 |
| 总延迟时间 | Total latency | 所有成功请求的延迟时间总和 | 所有成功请求的延迟时间之和 |
| 平均延迟 | Average latency | 从发送请求到接收完整响应的平均时间 | 总延迟时间 / 成功请求数 |
| 平均首token时间 | Average time to first token | 从发送请求到接收到第一个响应标记的平均时间 | 总首chunk延迟 / 成功请求数 |
| 平均每输出token时间 | Average time per output token | 生成每个输出标记所需的平均时间 | 总每输出token时间 / 成功请求数 |
| 平均输入token数 | Average input tokens per request | 每个请求的平均输入标记数 | 总输入token数 / 成功请求数 |
| 平均输出token数 | Average output tokens per request | 每个请求的平均输出标记数 | 总输出token数 / 成功请求数 |
| 平均数据包延迟 | Average package latency | 接收每个数据包的平均延迟时间 | 总数据包时间 / 总数据包数 |
| 平均每请求数据包数 | Average package per request | 每个请求平均接收的数据包数量 | 总数据包数 / 成功请求数 |

**百分位指标 (Percentile)**

> 以单个请求为单位进行统计，数据被分为100个相等部分，第n百分位表示n%的数据点在此值之下。

| 指标 | 英文名称 | 解释 |
|------|----------|------|
| 首次生成token时间 | TTFT (Time to First Token) | 从发送请求到生成第一个token的时间（以秒为单位） |
| 输出token间时延 | ITL (Inter-token Latency) | 生成每个输出token所需的时间（以秒为单位） |
| 延迟时间 | Latency | 从发送请求到接收完整响应的时间（以秒为单位） |
| 输入token数 | Input tokens | 请求中输入的token数量 |
| 输出token数 | Output tokens | 响应中生成的token数量 |
| 吞吐量 | Throughput | 每秒输出的token数量（tokens/s） |


## 可视化测试结果

### 使用WandB进行可视化测试结果

请先安装wandb，并获取对应的[API Key](https://wandb.ai/settings)：
```bash
pip install wandb
```

在评测启动时，额外添加以下参数，即可将测试结果上传wandb server并进行可视化：
```bash
# ...
--wandb-api-key 'wandb_api_key'
--name 'name_of_wandb_log'
```  

例如：

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)

### 使用SwanLab进行可视化测试结果

请先安装SwanLab，并获取对应的[API Key](https://swanlab.cn/space/~/settings)

```bash
pip install swanlab
```

在评测启动时，额外添加以下参数，即可将测试结果上传swanlab server并进行可视化：
```bash
# ...
--swanlab-api-key 'swanlab_api_key'
--name 'name_of_swanlab_log'
```

例如：

![swanlab sample](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/swanlab.png)

如果希望仅[使用SwanLab本地看板模式](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html)，先安装swanlab离线看板：

```bash
pip install 'swanlab[dashboard]'
```

再通过设置如下参数：

```bash
--swanlab-api-key local
```

并通过`swanlab watch <日志路径>`打开本地可视化看板。
