# 快速上手

EvalScope 是一个为大模型设计的评测框架，旨在提供简单易用、功能全面的评测流程。本指南将引导您完成从简单到复杂的各类评测任务，帮助您快速上手。

## 开始第一次评测

您可以通过命令行或 Python 代码两种方式，在指定数据集上使用默认配置评测一个模型。

### 方式1：使用命令行

在任意路径下执行 `evalscope eval` 命令即可启动评测。这是最推荐的入门方式。

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```

**基本参数说明:**
- `--model`: 指定模型的 ModelScope ID (如 `Qwen/Qwen2.5-0.5B-Instruct`) 或本地路径。
- `--datasets`: 指定一个或多个数据集名称，以空格分隔。支持的数据集请参考[数据集列表](./supported_dataset/index.md)。
- `--limit`: 每个数据集最多评测的样本数，便于快速验证。若不设置，则评测全量数据。

### 方式2：使用 Python 代码

通过调用 `run_task` 函数并传入 `TaskConfig` 配置，可以在 Python 环境中运行评测。配置可以是 `TaskConfig` 对象、Python 字典，或 `yaml`/`json` 文件路径。

::::{tab-set}
:::{tab-item} 使用 `TaskConfig` 对象
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['gsm8k', 'arc'],
    limit=5
)

run_task(task_cfg=task_cfg)
```
:::
:::{tab-item} 使用 Python 字典
```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'datasets': ['gsm8k', 'arc'],
    'limit': 5
}

run_task(task_cfg=task_cfg)
```
:::
:::{tab-item} 使用 YAML 文件
```{code-block} yaml
:caption: config.yaml

model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
  - arc
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
:::
:::{tab-item} 使用 JSON 文件
```{code-block} json
:caption: config.json

{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "datasets": ["gsm8k", "arc"],
    "limit": 5
}
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.json")
```
:::
::::

### 查看评测结果

评测完成后，终端会打印出如下格式的得分报告：

```text
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Model Name            | Dataset Name   | Metric Name     | Category Name   | Subset Name   |   Num |   Score |
+=======================+================+=================+=================+===============+=======+=========+
| Qwen2.5-0.5B-Instruct | gsm8k          | AverageAccuracy | default         | main          |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Easy      |     5 |     0.8 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
| Qwen2.5-0.5B-Instruct | ai2_arc        | AverageAccuracy | default         | ARC-Challenge |     5 |     0.4 |
+-----------------------+----------------+-----------------+-----------------+---------------+-------+---------+
```

```{seealso}
除了命令行报告，您还可以使用我们的可视化工具来深入分析评测结果。

参考：[评测结果可视化](visualization.md)
```

## 进阶用法

### 自定义评测参数

**示例1. 调整模型加载和推理生成参数** 

您可以通过传入 JSON 格式的字符串来精细化控制模型加载、推理生成和数据集配置。

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
 --generation-config '{"do_sample":true,"temperature":0.6,"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

**常用参数说明:**
- `--model-args`: 模型加载参数，例如 `revision` (版本), `precision` (精度), `device_map` (设备映射)。
- `--generation-config`: 推理生成参数，例如 `do_sample` (采样), `temperature` (温度), `max_tokens` (最大长度)。
- `--dataset-args`: 数据集专属参数，以数据集名称为键。例如 `few_shot_num` (少样本数量)。

**示例2. 调整结果聚合方式**

对于需要多次生成的任务（如数学题），可以通过 `--repeats` 参数指定重复生成次数k，并使用 `dataset-args` 调整结果聚合方式，例如 `mean_and_vote_at_k`、`mean_and_pass_at_k`、`mean_and_pass^k`。

```shell
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k \
 --limit 10 \
 --dataset-args '{"gsm8k": {"aggregation": "mean_and_vote_at_k"}}' \
 --repeats 5
```


```{seealso}
查看所有可配置项，请参考：[全部参数说明](parameters.md)
```

### 评测在线模型 API

EvalScope 支持评测兼容 OpenAI API 格式的模型服务。只需指定服务地址、API Key，并将 `eval-type` 设置为 `openai_api`。

**1. 启动模型服务**

以 vLLM 为例，启动一个模型服务：
```shell
# 请先安装 vLLM: pip install vllm
export VLLM_USE_MODELSCOPE=True
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen2.5 \
  --trust-remote-code \
  --port 8801
```

**2. 运行评测**

使用以下命令评测该 API 服务：
```shell
evalscope eval \
 --model qwen2.5 \
 --api-url http://127.0.0.1:8801/v1 \
 --api-key EMPTY \
 --eval-type openai_api \
 --datasets gsm8k \
 --limit 10
```

### 使用裁判模型进行评估

对于某些主观或开放式任务（如 `simple_qa`），可以指定一个强大的“裁判模型”来评估目标模型的输出。

```python
import os
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='qwen2.5-7b-instruct',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type=EvalType.SERVICE,
    datasets=['chinese_simpleqa'],
    limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'qwen2.5-72b-instruct',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
    }
)

run_task(task_cfg=task_cfg)
```

```{seealso}
关于裁判模型的详细配置，请参考：[裁判模型参数](./parameters.md#judge参数)
```

### 离线评测

在无网络环境下，您可以使用本地缓存的模型和数据集进行评测。

**1. 准备本地数据集**

数据集默认托管在[ModelScope](https://modelscope.cn/datasets)上，加载需要联网。如果是无网络环境，可以使用本地数据集和模型，流程如下：

1) 首先查看想要使用的数据集在modelscope上的ID：在[支持的数据集](./supported_dataset/index.md)列表中查找到需要使用的数据集的**数据集ID**，例如`mmlu_pro`的ID为 `modelscope/MMLU-Pro`。
2) 使用modelscope命令下载数据集：点击“数据集文件”tab -> 点击“下载数据集” -> 复制命令行
```bash
# 下载数据集
modelscope download --dataset modelscope/MMLU-Pro --local_dir ./data/mmlu_pro
```
3) 使用目录`./data/mmlu_pro`作为`local_path`参数的值传入即可。

**2. 准备本地模型**

模型文件托管在ModelScope Hub端，需要联网加载，当需要在离线环境创建评测任务时，可提前将模型下载到本地：

例如使用modelscope命令下载[Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)模型到本地：
```bash
modelscope download --model modelscope/Qwen2.5-0.5B-Instruct --local_dir ./model/qwen2.5
```

```{seealso}
更多下载选项，请参考 [ModelScope下载指南](https://modelscope.cn/docs/models/download)。
```

**3. 运行离线评测**

在评测命令中，将 `--model` 指向本地模型路径，并通过 `--dataset-args` 指定数据集的 `local_path`。

```shell
evalscope eval \
 --model ./model/qwen2.5 \
 --datasets mmlu_pro \
 --dataset-args '{"mmlu_pro": {"local_path": "./data/mmlu_pro"}}' \
 --limit 10
```

## 从 v0.1.x 迁移

如果您之前使用的是 v0.1.x 版本，升级到 v1.0+ 后请注意以下主要变化：

1.  **本地数据集**：不再支持 `zip` 压缩包格式。请参考[离线评测](#离线评测)指引，使用 `local_path` 参数加载本地数据集。
2.  **可视化兼容性**：v1.0 的输出报告格式已更新，旧版报告与新版可视化工具不兼容。
3.  **推理参数**：模型推理参数 `n` 已被移除，请改用 `repeats` 参数来控制重复生成次数。
4.  **评测流程控制**：`stage` 参数已移除。新增 `rerun_review` 参数，用于在 `use_cache=True` 时强制重新执行评分步骤。
5.  **数据集名称**：`gpqa` 数据集已更名为 `gpqa_diamond`，且不再需要手动指定 `subset_list`。

## 常见问题

在评测过程中遇到问题？我们为您整理了一份常见问题列表，希望能帮助您解决问题。

```{seealso}
参考：[常见问题解答](faq.md)
```