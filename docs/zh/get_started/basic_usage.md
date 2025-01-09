# 基本使用

## 简单评测
在指定的若干数据集上使用默认配置评测某个模型，本框架支持两钟启动评测任务的方式：使用命令行启动或使用Python代码启动评测任务。

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
- `--datasets`: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动从modelscope下载，支持的数据集参考[数据集列表](./supported_dataset.md#支持的数据集)
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
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --model-args revision=master,precision=torch.float16,device_map=auto \
 --generation-config do_sample=true,temperature=0.5 \
 --dataset-args '{"gsm8k": {"few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

### 参数说明
- `--model-args`: 模型加载参数，以逗号分隔，`key=value`形式，默认参数：
  - `revision`: 模型版本，默认为`master`
  - `precision`: 模型精度，默认为`auto`
  - `device_map`: 模型分配设备，默认为`auto`
- `--generation-config`: 生成参数，以逗号分隔，`key=value`形式，默认参数：
  - `do_sample`: 是否使用采样，默认为`false`
  - `max_length`: 最大长度，默认为2048
  - `max_new_tokens`: 生成最大长度，默认为512
- `--dataset-args`: 评测数据集的设置参数，以`json`格式传入，key为数据集名称，value为参数，注意需要跟`--datasets`参数中的值一一对应：
  - `few_shot_num`: few-shot的数量
  - `few_shot_random`: 是否随机采样few-shot数据，如果不设置，则默认为`true`

```{seealso}
参考：[全部参数说明](parameters.md)
```

### 输出结果

```text
+-----------------------+-----------------+
| Model                 | gsm8k           |
+=======================+=================+
| Qwen2.5-0.5B-Instruct | (gsm8k/acc) 0.2 |
+-----------------------+-----------------+
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
 --api-url http://127.0.0.1:8801/v1/chat/completions \
 --api-key EMPTY \
 --eval-type service \
 --datasets gsm8k \
 --limit 10
```


## 离线评测

数据集默认托管在[ModelScope](https://modelscope.cn/datasets)上，加载需要联网。如果是无网络环境，可以使用本地数据集，流程如下：

假如当前本地工作路径为 `/path/to/workdir`。

### 下载数据集到本地
执行以下命令：
```shell
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```
解压后的数据集在：`/path/to/workdir/data` 目录下，该目录在后续步骤将会作为`local_path`参数的值传入


### 下载模型到本地
模型文件托管在ModelScope Hub端，需要联网加载，当需要在离线环境创建评测任务时，可提前将模型下载到本地：

例如使用Git下载Qwen2.5-0.5B-Instruct模型到本地：
```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git
```

```{seealso}
[ModelScope下载模型指南](https://modelscope.cn/docs/models/download)
```

### 执行评测任务
运行下面的命令进行评测，传入本地数据集路径和模型路径，注意`local_path`需要跟`--datasets`参数中的值一一对应：

```shell
evalscope eval \
 --model /path/to/workdir/Qwen2.5-0.5B-Instruct \
 --datasets arc \
 --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' \
 --limit 10
```