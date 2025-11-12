(opencompass)=

# OpenCompass

为便于使用OpenCompass 评测后端，我们基于OpenCompass源码做了定制，命名为`ms-opencompass`，该版本在原版基础上对评测任务的配置和执行做了一些优化，并支持pypi安装方式，使得用户可以通过EvalScope发起轻量化的OpenCompass评测任务。同时，我们先期开放了基于OpenAI API格式的接口评测任务，您可以使用[ms-swift](https://github.com/modelscope/swift)、[vLLM](https://github.com/vllm-project/vllm)、[LMDeploy](https://github.com/InternLM/lmdeploy)、[Ollama](https://ollama.ai/)等模型服务，来拉起模型推理服务。

## 1. 环境准备
```shell
# 安装opencompass依赖
pip install evalscope[opencompass] -U
```

## 2. 数据准备


````{note}
有以下两种方式下载数据集

数据集的详细信息可以参考[OpenCompass数据集列表](../../get_started/supported_dataset/other/opencompass.md)

您可以使用以下方式，来查看支持的数据集的名称列表：
```python
from evalscope.backend.opencompass import OpenCompassBackendManager
# 显示支持的数据集名称列表
OpenCompassBackendManager.list_datasets()

>>> ['summedits', 'humaneval', 'lambada', 
'ARC_c', 'ARC_e', 'CB', 'C3', 'cluewsc', 'piqa',
 'bustm', 'storycloze', 'lcsts', 'Xsum', 'winogrande', 
 'ocnli', 'AX_b', 'math', 'race', 'hellaswag', 
 'WSC', 'eprstmt', 'siqa', 'agieval', 'obqa',
 'afqmc', 'GaokaoBench', 'triviaqa', 'CMRC', 
 'chid', 'gsm8k', 'ceval', 'COPA', 'ReCoRD', 
 'ocnli_fc', 'mbpp', 'csl', 'tnews', 'RTE', 
 'cmnli', 'AX_g', 'nq', 'cmb', 'BoolQ', 'strategyqa', 
 'mmlu', 'WiC', 'MultiRC', 'DRCD', 'cmmlu']
```
````

::::{tab-set}
:::{tab-item} 设置环境变量自动下载（推荐）
支持从 ModelScope 自动下载数据集，要启用此功能，请设置环境变量：
```shell
export DATASET_SOURCE=ModelScope
```
以下数据集在使用时将自动下载：

| 名称               | 名称               |
|--------------------|--------------------|
| humaneval          | AGIEval            |
| triviaqa           | gsm8k              |
| commonsenseqa      | nq                 |
| tydiqa             | race               |
| strategyqa         | siqa               |
| cmmlu              | mbpp               |
| lambada            | hellaswag          |
| piqa               | ARC                |
| ceval              | BBH                |
| math               | xstory_cloze       |
| LCSTS              | summedits          |
| Xsum               | GAOKAO-BENCH      |
| winogrande         | OCNLI              |
| openbookqa         | cmnli              |


:::

:::{tab-item} 使用链接下载
```shell
# ModelScope下载
wget -O eval_data.zip https://www.modelscope.cn/datasets/swift/evalscope_resource/resolve/master/eval.zip

# 或使用github下载
wget -O eval_data.zip https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip

# 解压
unzip eval_data.zip
```
包含的数据集有：

| 名称                       | 名称                       | 名称                       |
|----------------------------|----------------------------|----------------------------|
| obqa                       | AX_b                       | siqa                       |
| nq                         | mbpp                       | winogrande                 |
| mmlu                       | BoolQ                      | cluewsc                    |
| ocnli                      | lambada                    | CMRC                       |
| ceval                      | csl                        | cmnli                      |
| bbh                        | ReCoRD                     | math                       |
| humaneval                  | eprstmt                    | WSC                        |
| storycloze                 | MultiRC                    | RTE                        |
| chid                       | gsm8k                      | AX_g                       |
| bustm                      | afqmc                      | piqa                       |
| lcsts                      | strategyqa                 | Xsum                       |
| agieval                    | ocnli_fc                   | C3                         |
| tnews                      | race                       | triviaqa                   |
| CB                         | WiC                        | hellaswag                  |
| summedits                  | GaokaoBench                | ARC_e                      |
| COPA                       | ARC_c                      | DRCD                       |

总大小约1.7GB，下载并解压后，将数据集文件夹（即data文件夹）放置在当前工作路径下。
:::
::::


## 3. 部署模型服务
OpenCompass 评测后端使用统一的OpenAI API调用来进行评测，因此我们需要进行模型部署。

下面介绍四种方式部署模型服务：
::::{tab-set}

:::{tab-item} vLLM 部署（推荐）
参考 [vLLM 教程](https://docs.vllm.ai/en/latest/index.html)。 

[支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)

**安装vLLM**
```shell
pip install vllm -U
```

**部署模型服务**
```shell
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-0.5B-Instruct --port 8000 --served-model-name Qwen2-0.5B-Instruct
```
:::

:::{tab-item} ms-swift部署 

使用ms-swift部署模型服务，具体可参考：[ms-swift部署指南](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%8E%A8%E7%90%86%E5%92%8C%E9%83%A8%E7%BD%B2.html#id1)。

**安装ms-swift**
```shell
pip install ms-swift -U
```

**部署模型服务**
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

<details><summary>ms-swift v2.x</summary>

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-0_5b-instruct --port 8000
```

</details>

:::

:::{tab-item} LMDeploy 部署
参考 [LMDeploy 教程](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/api_server_vl.md)。

**安装LMDeploy**
```shell
pip install lmdeploy -U
```

**部署模型服务**
```shell
LMDEPLOY_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server Qwen/Qwen2-0.5B-Instruct --server-port 8000
```
:::

:::{tab-item} Ollama 部署

```{note}
Ollama 对于 OpenAI API 的支持目前处于实验性状态，本教程仅提供示例，请根据实际情况修改。
```

参考 [Ollama 教程](https://github.com/ollama/ollama/blob/main/README.md#quickstart)。

**安装Ollama**
```shell
# Linux 系统
curl -fsSL https://ollama.com/install.sh | sh
```

**启动Ollama**
```shell
# 默认端口为 11434
ollama serve
```

```{tip}
若使用`ollama pull`拉取模型，可跳过以下创建模型的步骤；若使用`ollama import`导入模型，则需要手动创建模型配置文件。
```

**创建模型配置文件 `Modelfile`**

[支持的模型格式](https://github.com/ollama/ollama/blob/main/docs/import.md)
```text
# 模型路径
FROM models/Meta-Llama-3-8B-Instruct

# 温度系数
PARAMETER temperature 1

# system prompt
SYSTEM """
You are a helpful assistant.
"""
```

**创建模型**

会将模型自动转为ollama支持的格式，同时支持多种量化方式。
```shell
ollama create llama3 -f ./Modelfile
```
:::

::::


## 4. 模型评测

### 配置文件
有如下三种方式编写配置文件：
::::{tab-set}
:::{tab-item} Python 字典
```python
task_cfg_dict = dict(
    eval_backend='OpenCompass',
    eval_config={
        'datasets': ["mmlu", "ceval",'ARC_c', 'gsm8k'],
        'models': [
            {'path': 'Qwen2-0.5B-Instruct', 
            'openai_api_base': 'http://127.0.0.1:8000/v1/chat/completions', 
            'is_chat': True,
            'batch_size': 16},
        ],
        'work_dir': 'outputs/qwen2_eval_result',
        'limit': 10,
        },
    )
```
:::

:::{tab-item} yaml 配置文件
```{code-block} yaml
:caption: eval_openai_api.yaml

eval_backend: OpenCompass
eval_config:
  datasets:
    - mmlu
    - ceval
    - ARC_c
    - gsm8k
  models:
    - openai_api_base: http://127.0.0.1:8000/v1/chat/completions
      path: Qwen2-0.5B-Instruct                                   
      temperature: 0.0
```
:::

:::{tab-item} json 配置文件
```{code-block} json
:caption: eval_openai_api.json
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
        "path": "Qwen2-0.5B-Instruct",
        "openai_api_base": "http://127.0.0.1:8000/v1/chat/completions",
        "temperature": 0.0
      }
    ]
  }
}

```
:::
::::

#### 参数说明

- `eval_backend`：默认值为 `OpenCompass`，表示使用 OpenCompass 评测后端
- `eval_config`：字典，包含以下字段：
  - `datasets`：列表，参考[目前支持的数据集](#2-数据准备)
  - `models`：字典列表，每个字典必须包含以下字段：
    - `path`：OpenAI API 请求模型名称。
      - 若使用`ms-swift`部署，设置为 `--model_type` 的值；
      - 若使用 `vLLM` 或 `LMDeploy` 部署模型，则设置为对应的 model ID；
      - 若使用 `Ollama` 部署模型，则设置为 `model_name`，使用`ollama list`命令查看。
    - `openai_api_base`：OpenAI API 的URL。
    - `is_chat`：布尔值，设置为 `True` 表示聊天模型，设置为 `False` 表示基础模型。
    - `key`：模型 API 的 OpenAI API 密钥，默认值为 `EMPTY`。
  - `work_dir`：字符串，保存评测结果、日志和摘要的目录。默认值为 `outputs/default`。
  - `limit`： 可以是 `int`、`float` 或 `str`，例如 5、5.0 或 `'[10:20]'`。默认值为 `None`，表示运行所有示例。

有关其他可选属性，请参考 `opencompass.cli.arguments.ApiModelConfig`。

### 运行脚本
配置好配置文件后，运行以下脚本即可

```{code-block} python
:caption: example_eval_openai_api.py

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_eval():
    # 选项 1: python 字典
    task_cfg = task_cfg_dict

    # 选项 2: yaml 配置文件
    # task_cfg = 'eval_openai_api.yaml'

    # 选项 3: json 配置文件
    # task_cfg = 'eval_openai_api.json'

    run_task(task_cfg=task_cfg)

    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()
```
运行以下命令：
```shell
python eval_openai_api.py
```


可以看到最终输出如下：

```text
dataset                                 version    metric         mode    Qwen2-0.5B-Instruct
--------------------------------------  ---------  -------------  ------  ---------------------
--------- 考试 Exam ---------           -          -              -       -
ceval                                   -          naive_average  gen     30.00
agieval                                 -          -              -       -
mmlu                                    -          naive_average  gen     42.28
GaokaoBench                             -          -              -       -
ARC-c                                   1e0de5     accuracy       gen     60.00
--------- 语言 Language ---------       -          -              -       -
WiC                                     -          -              -       -
summedits                               -          -              -       -
winogrande                              -          -              -       -
flores_100                              -          -              -       -
--------- 推理 Reasoning ---------      -          -              -       -
cmnli                                   -          -              -       -
ocnli                                   -          -              -       -
ocnli_fc-dev                            -          -              -       -
AX_b                                    -          -              -       -
AX_g                                    -          -              -       -
strategyqa                              -          -              -       -
math                                    -          -              -       -
gsm8k                                   1d7fe4     accuracy       gen     40.00
...
```
