# 基本使用

## 简单评测
在指定的若干数据集上使用默认配置评测某个模型，本框架支持两种启动评测任务的方式：使用命令行启动或使用Python代码启动评测任务。

### 方式1. 使用命令行

::::{tab-set}
:::{tab-item} 使用`eval`命令

在任意路径下执行`eval`命令：
```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```
:::

:::{tab-item} 运行`run.py`

在`evalscope`根目录下执行：
```bash
python evalscope/run.py \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k arc \
 --limit 5
```
:::
::::

### 方式2. 使用Python代码

使用python代码进行评测时需要用`run_task`函数提交评测任务，传入一个`TaskConfig`作为参数，也可以为python字典、yaml文件路径或json文件路径，例如：

::::{tab-set}
:::{tab-item} 使用Python 字典

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

:::{tab-item} 使用`TaskConfig`

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

:::{tab-item} 使用`yaml`文件

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

:::{tab-item} 使用`json`文件

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


### 基本参数说明
- `--model`: 指定了模型在[ModelScope](https://modelscope.cn/)中的`model_id`，可自动下载，例如[Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/summary)；也可使用模型的本地路径，例如`/path/to/model`
- `--datasets`: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动从modelscope下载，支持的数据集参考[数据集列表](./supported_dataset/index.md)
- `--limit`: 每个数据集最大评测数据量，不填写则默认为全部评测，可用于快速验证


### 输出结果
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


## 复杂评测
若想进行更加自定义的评测，例如自定义模型参数，或者数据集参数，可以使用以下命令，启动评测方式与简单评测一致，下面展示了使用`eval`命令启动评测：

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
 --generation-config '{"do_sample":true,"temperature":0.6,"max_tokens":512,"chat_template_kwargs":{"enable_thinking": false}}' \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

### 参数说明
- `--model-args`: 模型加载参数，以json字符串格式传入：
  - `revision`: 模型版本
  - `precision`: 模型精度
  - `device_map`: 模型分配设备
- `--generation-config`: 生成参数，以json字符串格式传入，将解析为字典：
  - `do_sample`: 是否使用采样
  - `temperature`: 生成温度
  - `max_tokens`: 生成最大长度
  - `chat_template_kwargs`: 模型推理模板参数
- `--dataset-args`: 评测数据集的设置参数，以json字符串格式传入，key为数据集名称，value为参数，注意需要跟`--datasets`参数中的值一一对应：
  - `few_shot_num`: few-shot的数量
  - `few_shot_random`: 是否随机采样few-shot数据，如果不设置，则默认为`true`

```{seealso}
参考：[全部参数说明](parameters.md)
```

### 输出结果

```text
+------------+-----------+-----------------+----------+-------+---------+---------+
| Model      | Dataset   | Metric          | Subset   |   Num |   Score | Cat.0   |
+============+===========+=================+==========+=======+=========+=========+
| Qwen3-0.6B | gsm8k     | AverageAccuracy | main     |    10 |     0.3 | default |
+------------+-----------+-----------------+----------+-------+---------+---------+ 
```

## 模型API服务评测

指定模型API服务地址(api_url)和API Key(api_key)，评测部署的模型API服务，*此时`eval-type`参数必须指定为`service`*：

例如使用[vLLM](https://github.com/vllm-project/vllm)拉起模型服务：
```shell
export VLLM_USE_MODELSCOPE=True && python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --served-model-name qwen2.5 --trust_remote_code --port 8801
```
然后使用以下命令评测模型API服务：
```shell
evalscope eval \
 --model qwen2.5 \
 --api-url http://127.0.0.1:8801/v1 \
 --api-key EMPTY \
 --eval-type service \
 --datasets gsm8k \
 --limit 10
```

## 使用裁判模型

在评测时，可以使用裁判模型对模型的输出进行评估，此外有些数据集需要使用裁判模型进行评测，例如`simple_qa`数据集，使用以下命令启动评测：

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='qwen2.5-7b-instruct',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key= os.getenv('DASHSCOPE_API_KEY'),
    eval_type=EvalType.SERVICE,
    datasets=[
        # 'simple_qa',
        'chinese_simpleqa',
    ],
    eval_batch_size=5,
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
参考：[裁判模型参数](./parameters.md#judge参数)
```

## 离线评测

数据集默认托管在[ModelScope](https://modelscope.cn/datasets)上，加载需要联网。如果是无网络环境，可以使用本地数据集和模型，流程如下：

### 使用本地数据集

1. 首先查看想要使用的数据集在modelscope上的地址：数据集地址参考[支持的数据集](./supported_dataset/index.md)中，找到数据集的**数据集ID**，例如[MMLU-Pro](https://modelscope.cn/datasets/modelscope/MMLU-Pro/summary)
2. 使用modelscope命令下载数据集：点击“数据集文件”tab -> 点击“下载数据集” -> 复制命令行 
```bash
modelscope download --dataset modelscope/MMLU-Pro --local_dir ./data/mmlu_pro
```
3. 使用目录`./data/mmlu_pro`作为`local_path`参数的值传入即可。

### 使用本地模型
模型文件托管在ModelScope Hub端，需要联网加载，当需要在离线环境创建评测任务时，可提前将模型下载到本地：

例如使用modelscope命令下载[Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)模型到本地：
```bash
modelscope download --model modelscope/Qwen2.5-0.5B-Instruct --local_dir ./model/qwen2.5
```

```{seealso}
[ModelScope下载模型指南](https://modelscope.cn/docs/models/download)
```

### 执行评测任务
运行下面的命令进行评测，传入本地数据集路径和模型路径，注意`local_path`需要跟`--datasets`参数中的值一一对应：

```shell
evalscope eval \
 --model ./model/qwen2.5 \
 --datasets mmlu_pro \
 --dataset-args '{"mmlu_pro": {"local_path": "./data/mmlu_pro"}}' \
 --limit 10
```

## 切换到v1.0版本

如果你之前使用的是 v0.1x 版本，想切换到使用 v1.0 版本，需要注意以下变化：
1. 使用zip下载的评测数据集不再支持，使用本地数据集请[参考](#使用本地数据集)
2. 由于评测输出格式的变化，EvalScope 可视化功能不兼容1.0之前版本，需要使用1.0版本输出的report才能正常可视化。
3. 由于数据集格式的变化，EvalScope的数据集合功能不支持1.0之前版本创建的数据集，需要使用1.0版本重新运行数据集合创建。
4. 模型推理参数中的`n`参数不再支持，更改为`repeats`
5. `stage`参数已移除，新增`rerun_review`参数，来控制在指定了`use_cache`的情况下是否重新运行评测。