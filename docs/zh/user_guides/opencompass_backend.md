# OpenCompass 评测后端

为便于使用OpenCompass 评测后端，我们基于OpenCompass源码做了定制，命名为`ms-opencompass`，该版本在原版基础上对评估任务的配置和执行做了一些优化，并支持pypi安装方式，使得用户可以通过EvalScope发起轻量化的OpenCompass评估任务。同时，我们先期开放了基于OpenAI API格式的接口评估任务，您可以使用[ms-swift](https://github.com/modelscope/swift) 部署模型服务，其中，[swift deploy](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm)支持使用[vLLM](https://github.com/vllm-project/vllm)拉起模型推理服务。

## 1. 环境准备
```shell
# 安装opencompass依赖
pip install evalscope[opencompass] -U
```

## 2. 数据准备

<!-- ````{note}
您可以使用以下方式，来查看数据集的名称列表：
```python
from evalscope.backend.opencompass import OpenCompassBackendManager
print(f'All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')
```
```` -->

````{note}
有以下三种方式下载数据集，其中自动下载数据集方式支持的数据集少于手动下载数据集方式，请按需使用。

数据集的详细信息可以参考[OpenCompass数据集列表](https://hub.opencompass.org.cn/home)

您可以使用以下方式，来查看数据集的名称列表：
```python
from evalscope.backend.opencompass import OpenCompassBackendManager
print(f'All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')
```
````

`````{tabs}
````{tab} 设置环境变量自动下载（推荐）
支持从 ModelScope 自动下载数据集，要启用此功能，请设置环境变量：
```shell
export DATASET_SOURCE=ModelScope
```
以下数据集在使用时将自动下载：

```text
humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
```
````

````{tab} 使用ModelScope链接下载
```shell
# 下载
wget -O eval_data.zip https://www.modelscope.cn/datasets/swift/evalscope_resource/resolve/master/eval.zip
# 解压
unzip eval_data.zip
```
包含的数据集有：
```text
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada', 'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze', 'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval', 'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench', 'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```

总大小约1.7GB，下载并解压后，将数据集文件夹（即data文件夹）放置在当前工作路径下。
````

````{tab} 使用github链接下载
```shell
# 下载
wget -O eval_data.zip https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
# 解压
unzip eval_data.zip
```
包含的数据集有：
```text
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada', 'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze', 'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval', 'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench', 'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
总大小约1.7GB，下载并解压后，将数据集文件夹（即data文件夹）放置在当前工作路径下。
````

`````


## 3. 模型推理服务
OpenCompass 评测后端使用统一的OpenAI API调用来进行评估，因此我们需要进行模型部署。下面我们使用ms-swift部署模型服务，具体可参考：[ms-swift部署指南](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm)

### 安装ms-swift
```shell
pip install ms-swift -U
```

### 部署模型服务
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-0_5b-instruct --port 8000

# 启动成功日志
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```


## 4. 模型评估

### 配置文件
有如下三种方式编写配置文件：
`````{tabs}
````{tab} Python 字典
```python
task_cfg_dict = dict(
    eval_backend='OpenCompass',
    eval_config={
        'datasets': ["mmlu", "ceval",'ARC_c', 'gsm8k'],
        'models': [
            {'path': 'qwen2-0_5b-instruct', 
            'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions', 
            'is_chat': True,
            'batch_size': 16},
        ],
        'work_dir': 'outputs/qwen2_eval_result',
        'limit': 10,
        },
    )
```
````

````{tab} yaml 配置文件
eval_swift_openai_api.yaml
```yaml
eval_backend: OpenCompass
eval_config:
  datasets:
    - mmlu
    - ceval
    - ARC_c
    - gsm8k
  models:
    - openai_api_base: http://127.0.0.1:8000/v1/chat/completions
      path: qwen2-0_5b-instruct                                   
      temperature: 0.0
```
````

````{tab} json 配置文件
eval_swift_openai_api.json
```json
{
  "eval_backend": "OpenCompass",
  "eval_config": {
    "datasets": [
      "mmlu",
      "ceval",
      "ARC_c",
      "gsm8k"
    ],
    "models": [
      {
        "path": "qwen2-0_5b-instruct",
        "openai_api_base": "http://127.0.0.1:8000/v1/chat/completions",
        "temperature": 0.0
      }
    ]
  }
}

```
````
`````
#### 参数说明

- `eval_backend`：默认值为 `OpenCompass`，表示使用 OpenCompass 评测后端
- `eval_config`：字典，包含以下字段：
  - `datasets`：列表，参考[目前支持的数据集](#2-数据准备)
  - `models`：字典列表，每个字典必须包含以下字段：
    - `path`：重用命令行 `swift deploy` 中的 `--model_type` 的值
    - `openai_api_base`：OpenAI API 的URL，即 Swift 模型服务的 URL
    - `is_chat`：布尔值，设置为 `True` 表示聊天模型，设置为 `False` 表示基础模型
    - `key`：模型 API 的 OpenAI API 密钥，默认值为 `EMPTY`
  - `work_dir`：字符串，保存评估结果、日志和摘要的目录。默认值为 `outputs/default`
  - `limit`： 可以是 `int`、`float` 或 `str`，例如 5、5.0 或 `'[10:20]'`。默认值为 `None`，表示运行所有示例。

有关其他可选属性，请参考 `opencompass.cli.arguments.ApiModelConfig`。

### 运行脚本
配置好配置文件后，运行以下脚本即可

```python
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_swift_eval():
    # 选项 1: python 字典
    task_cfg = task_cfg_dict

    # 选项 2: yaml 配置文件
    # task_cfg = 'eval_swift_openai_api.yaml'

    # 选项 3: json 配置文件
    # task_cfg = 'eval_swift_openai_api.json'

    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_swift_eval()
```
或者运行以下命令：
```shell
python examples/example_eval_swift_openai_api.py
```


可以看到最终输出如下：
