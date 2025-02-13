# DeepSeek-R1类模型数学能力测试

随着DeepSeek-R1模型的爆火，越来越多的用户开始复现类似的模型，以提升模型的推理能力，目前也出现了许多惊艳的工作。但是，这些模型的推理能力是否真的有所提升呢？本文将通过数学能力测试来验证类DeepSeek-R1模型的推理能力。

本最佳实践将以DeepSeek-R1-Distill-Qwen-1.5B模型为例，使用共728道数学推理题目（与R1技术报告一致）进行演示，数据具体包括：

- [MATH-500](https://www.modelscope.cn/datasets/HuggingFaceH4/aime_2024) 500道数学推理题目
- [GPQA-Diamond](https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond/summary) 198道数学推理题目
- [AIME-2024](https://modelscope.cn/datasets/AI-ModelScope/AIME_2024) 30道数学推理题目

我们构建了这三个数据集的混合，放在[modelscope/R1-Distill-Math-Test](https://modelscope.cn/datasets/modelscope/R1-Distill-Math-Test)数据集中。

## 安装依赖

安装[EvalScope](https://github.com/modelscope/evalscope)模型评估框架：

由于框架在快速迭代中，pip包更新可能不及时，推荐通过源码安装：
```bash
git clone https://github.com/modelscope/evalscope.git
cd evalscope/
pip install -e '.[app,perf]'
```
或者通过pip安装：
```bash
pip install 'evalscope[app,perf]' -U
```

## 部署模型

由于R1类模型的输出一般很长（包含较长的思维链），需要使用推理框架部署模型来加速评测，这里我们介绍使用[vLLM](https://github.com/vllm-project/vllm)和[lmdeploy](https://github.com/InternLM/lmdeploy)，读者可以根据自己的需求选择其他部署工具。

**使用vLLM**:

```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  --served-model-name DeepSeek-R1-Distill-Qwen-1.5B --trust_remote_code --port 8801
```

或**使用lmdeploy**：

```bash
LMDEPLOY_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --model-name DeepSeek-R1-Distill-Qwen-1.5B --server-port 8801
```

### (可选) 测试推理服务性能

如果需要测试推理服务性能，可以使用`evalscope`的`perf`子命令，具体可参考[教程](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html)：

```bash
evalscope perf \
 --parallel 10 \
 --url http://127.0.0.1:8801/v1/chat/completions \
 --model DeepSeek-R1-Distill-Qwen-1.5B \
 --log-every-n-query 5 \
 --connect-timeout 6000 \
 --read-timeout 6000 \
 --api openai \
 --prompt '写一个科幻小说，不少于2000字，请开始你的表演' \
 -n 100
```

输出如下：

<details>

```text
Benchmarking summary:
+-----------------------------------+-------------------------------------------------------------------------+
| Key                               | Value                                                                   |
+===================================+=========================================================================+
| Time taken for tests (s)          | 92.66                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Number of concurrency             | 10                                                                      |
+-----------------------------------+-------------------------------------------------------------------------+
| Total requests                    | 100                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Succeed requests                  | 100                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Failed requests                   | 0                                                                       |
+-----------------------------------+-------------------------------------------------------------------------+
| Throughput(average tokens/s)      | 1727.453                                                                |
+-----------------------------------+-------------------------------------------------------------------------+
| Average QPS                       | 1.079                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average latency (s)               | 8.636                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average time to first token (s)   | 8.636                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average time per output token (s) | 0.00058                                                                 |
+-----------------------------------+-------------------------------------------------------------------------+
| Average input tokens per request  | 20.0                                                                    |
+-----------------------------------+-------------------------------------------------------------------------+
| Average output tokens per request | 1600.66                                                                 |
+-----------------------------------+-------------------------------------------------------------------------+
| Average package latency (s)       | 8.636                                                                   |
+-----------------------------------+-------------------------------------------------------------------------+
| Average package per request       | 1.0                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Expected number of requests       | 100                                                                     |
+-----------------------------------+-------------------------------------------------------------------------+
| Result DB path                    | outputs/20250213_103632/DeepSeek-R1-Distill-Qwen-1.5B/benchmark_data.db |
+-----------------------------------+-------------------------------------------------------------------------+

Percentile results:
+------------+----------+----------+-------------+--------------+---------------+----------------------+
| Percentile | TTFT (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Throughput(tokens/s) |
+------------+----------+----------+-------------+--------------+---------------+----------------------+
|    10%     |  5.4506  |   nan    |   5.4506    |      20      |     1011      |       183.7254       |
|    25%     |  6.1689  |   nan    |   6.1689    |      20      |     1145      |       184.9222       |
|    50%     |  9.385   |   nan    |    9.385    |      20      |     1741      |       185.5081       |
|    66%     | 11.0023  |   nan    |   11.0023   |      20      |     2048      |       185.8063       |
|    75%     | 11.0374  |   nan    |   11.0374   |      20      |     2048      |       186.1429       |
|    80%     |  11.047  |   nan    |   11.047    |      20      |     2048      |       186.3683       |
|    90%     |  11.075  |   nan    |   11.075    |      20      |     2048      |       186.5962       |
|    95%     |  11.147  |   nan    |   11.147    |      20      |     2048      |       186.7836       |
|    98%     | 11.1574  |   nan    |   11.1574   |      20      |     2048      |       187.4917       |
|    99%     | 11.1688  |   nan    |   11.1688   |      20      |     2048      |       197.4991       |
+------------+----------+----------+-------------+--------------+---------------+----------------------+
```

</details>

## 评测模型

### (可选) 自定义混合数据

使用EvalScope框架可以自定义混合数据集合，将其保存在本地或推送到modelscope上，参考[使用教程](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/collection/index.html)。

下面是一个例子，将`math_500`、`gpqa_diamond`和`aime24`三个数据集混合在一起，并放在了[modelscope/R1-Distill-Math-Test](https://modelscope.cn/datasets/modelscope/R1-Distill-Math-Test)数据集中，读者可跳过此步骤直接使用数据集：

```python
from evalscope.collections import WeightedSampler, CollectionSchema, DatasetInfo
from evalscope.utils.io_utils import dump_jsonl_data

# 定义数据混合schema
schema = CollectionSchema(name='DeepSeekDistill', datasets=[
            CollectionSchema(name='Math', datasets=[
                DatasetInfo(name='math_500', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
                DatasetInfo(name='gpqa', weight=1, task_type='math', tags=['en'],  args={'subset_list': ['gpqa_diamond'], 'few_shot_num': 0}),
                DatasetInfo(name='aime24', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
            ])
        ])

#  混合数据
mixed_data = WeightedSampler(schema).sample(100000)  # 设置一个较大的数目，确保所有数据都被采样到
# 保存数据到本地
dump_jsonl_data(mixed_data, 'test.jsonl')
```

### 配置评测任务

运行下面的python代码评测DeepSeek-R1-Distill-Qwen-1.5B模型在数学推理数据集上的表现：

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='DeepSeek-R1-Distill-Qwen-1.5B',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://127.0.0.1:8801/v1/chat/completions',  # 推理服务地址
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,   # 评测类型，SERVICE表示评测推理服务
    datasets=[
        'data_collection',  # 数据集名称(固定为data_collection表示使用混合数据集)
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'modelscope/R1-Distill-Math-Test'  # 数据集ID 或 数据集本地路径
        }
    },
    eval_batch_size=32,       # 发送请求的并发数
    generation_config={       # 模型推理配置
        'max_tokens': 20000,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 5                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
)

run_task(task_cfg=task_cfg)
```

**输出结果**:

**这里的计算指标是`AveragePass@1`，每个样本重复生成了5次，最终的评测结果是5次的平均值。** 由于模型生成时进行了采样，输出结果可能会有一定的波动。

```text
+-----------+--------------+---------------+-------+
| task_type | dataset_name | average_score | count |
+-----------+--------------+---------------+-------+
|   math    |   math_500   |    0.7832     | 2500  |
|   math    |     gpqa     |    0.3434     |  990  |
|   math    |    aime24    |      0.2      |  150  |
+-----------+--------------+---------------+-------+
```

----

如果只想运行单独其中的某些数据集，可以修改上述配置中的`datasets`和`dataset_args`参数，例如：

```python
datasets=[
    # 'math_500',  # 数据集名称
    'gpqa',
    'aime24'
],
dataset_args={ # EvalScope内置支持，无需指定数据集ID
    'math_500': {'few_shot_num': 0 } ,
    'gpqa': {'subset_list': ['gpqa_diamond'], 'few_shot_num': 0},
    'aime24': {'few_shot_num': 0}
},
```

## 评测结果可视化

EvalScope支持可视化结果，可以查看模型具体的输出。运行以下命令，可以启动可视化界面：
```bash
evalscope app
```

将输出如下链接内容：
```text
* Running on local URL:  http://0.0.0.0:7860
```

点击链接即可看到如下可视化界面，我们需要先选择评测报告然后点击加载：

<p align="center">
  <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/distill/score.png" alt="alt text" width="100%">
</p>


此外，选择对应的子数据集，我们也可以查看模型的输出内容，观察模型输出是否正确（或者是答案匹配是否存在问题）：

<p align="center">
  <img src="https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/distill/detail.png" alt="alt text" width="100%">
</p>