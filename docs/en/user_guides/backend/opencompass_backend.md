(opencompass)=
# OpenCompass

To facilitate the use of the OpenCompass evaluation backend, we have customized the OpenCompass source code and named it `ms-opencompass`. This version optimizes task configuration and execution and supports installation via PyPI, enabling users to initiate lightweight OpenCompass evaluation tasks through EvalScope. Additionally, we initially provide an interface for evaluation tasks based on the OpenAI API format. You can deploy model services using [ms-swift](https://github.com/modelscope/swift) , [vLLM](https://github.com/vllm-project/vllm) , [LMDeploy](https://github.com/InternLM/lmdeploy) , [Ollama](https://ollama.ai/), etc., to launch model inference services.

## 1. Environment Setup
```shell
# Install OpenCompass dependencies
pip install evalscope[opencompass] -U
```

## 2. Data Preparation
````{note}
There are two ways to download datasets. The automatic download method supports fewer datasets than the manual download method. Please use as needed. For detailed dataset information, refer to the [OpenCompass Dataset List](https://hub.opencompass.org.cn/home).
You can view the dataset name list using the following code:
```python
from evalscope.backend.opencompass import OpenCompassBackendManager
# list datasets
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
:::{tab-item} Automatically Download (Recommended)

Support for automatically downloading datasets from ModelScope. To enable this feature, please set the environment variable:
```shell
export DATASET_SOURCE=ModelScope
```
The following datasets will be downloaded automatically when used:
| Name               | Name               |
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

:::{tab-item} Download Using Links
```shell
# Download from ModelScope
wget -O eval_data.zip https://www.modelscope.cn/datasets/swift/evalscope_resource/resolve/master/eval.zip
# Or download from GitHub
wget -O eval_data.zip https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
# Unzip
unzip eval_data.zip
```
The included datasets are:

| Name                       | Name                       | Name                       |
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

The total size is approximately 1.7GB. After downloading and unzipping, place the dataset folder (i.e., the data folder) in the current working directory.
:::
::::

## 3. Model Inference Service
OpenCompass evaluation backend uses a unified OpenAI API call for assessment, so we need to deploy the model. 

Here are four ways to deploy model services:

::::{tab-set}
:::{tab-item} vLLM  (Recommended)
Refer to [vLLM Tutorial](https://docs.vllm.ai/en/latest/index.html) for more details.

[Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

**Install vLLM**
```shell
pip install vllm -U
```
**Deploy Model Service**
```shell
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-0.5B-Instruct --port 8000 --served-model-name Qwen2-0.5B-Instruct
```
:::

:::{tab-item} ms-swift
Use ms-swift to deploy model services. For more details, please refer to the: [ms-swift Deployment Guide](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html).

**Install ms-swift**
```shell
pip install ms-swift -U
```
**Deploy Model Service**

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

<details><summary>ms-swift v2.x</summary>

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-0_5b-instruct --port 8000
```

</details>

:::

:::{tab-item} LMDeploy
Refer to [LMDeploy Tutorial](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/api_server_vl.md) for more details.

**Install LMDeploy**
```shell
pip install lmdeploy -U
```

**Deploy Model Service**
```shell
LMDEPLOY_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server Qwen/Qwen2-0.5B-Instruct --server-port 8000
```
:::

:::{tab-item} Ollama
```{note}
Support for OpenAI API by Ollama is currently in an experimental state. This tutorial provides an example only; please modify it according to your actual situation.
```

Reference [Ollama Tutorial](https://github.com/ollama/ollama/blob/main/README.md#quickstart).

**Install Ollama**

```shell
# For Linux systems
curl -fsSL https://ollama.com/install.sh | sh
```

**Start Ollama**

```shell
# Default port is 11434
ollama server
```

```{tip}
If using `ollama pull` to fetch a model, you can skip the following steps for creating a model; if using `ollama import` to import a model, you will need to manually create a model configuration file.
```

**Create Model Configuration File `Modelfile`**

[Supported Model Formats](https://github.com/ollama/ollama/blob/main/docs/import.md)
```text
# Model path
FROM models/Meta-Llama-3-8B-Instruct

# Temperature coefficient
PARAMETER temperature 1

# System prompt
SYSTEM """
You are a helpful assistant.
"""
```
**Create Model**

The model will be automatically converted to a format supported by Ollama and supports multiple quantization methods.
```shell
ollama create llama3 -f ./Modelfile
```
:::

::::


## 4. Model Evaluation

### Configuration File
There are three ways to write the configuration file:
::::{tab-set}
:::{tab-item} Python Dictionary
```python
task_cfg_dict = dict(
    eval_backend='OpenCompass',
    eval_config={
        'datasets': ["mmlu", "ceval", 'ARC_c', 'gsm8k'],
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

:::{tab-item} YAML
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

:::{tab-item} JSON
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

#### Parameter Descriptions
- `eval_backend`: Default value is `OpenCompass`, indicating the use of the OpenCompass evaluation backend.
- `eval_config`: A dictionary containing the following fields:
  - `datasets`: A list, referring to the [currently supported datasets](#2-data-preparation).
  - `models`: A list of dictionaries, each dictionary must contain the following fields:
    - `path`: The model name for OpenAI API requests.
      - If deploying with `ms-swift`, set to the value of `--model_type`;
      - If deploying with `vLLM` or `LMDeploy`, set to model ID;
      - If deploying with `Ollama`, set to `model_name`, and use the `ollama list` command to check.
    - `openai_api_base`: The URL for the OpenAI API.
    - `is_chat`: Boolean value, set to `True` indicates a chat model; set to `False` indicates a base model.
    - `key`: The OpenAI API key for the model API, default value is `EMPTY`.
  - `work_dir`: A string specifying the directory to save evaluation results, logs, and summaries. Default value is `outputs/default`.
  - `limit`: Can be `int`, `float`, or `str`, for example, 5, 5.0, or `'[10:20]'`. Default value is `None`, which indicates all examples will be run.
For other optional attributes, refer to `opencompass.cli.arguments.ApiModelConfig`.

### Run Script
After configuring the configuration file, run the following script:
```{code-block} python
:caption: example_eval_openai_api.py

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_eval():
    # Option 1: Python dictionary
    task_cfg = task_cfg_dict

    # Option 2: YAML configuration file
    # task_cfg = 'eval_openai_api.yaml'

    # Option 3: JSON configuration file
    # task_cfg = 'eval_openai_api.json'
    
    run_task(task_cfg=task_cfg)
    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_eval()
```
Run the following command:
```shell
python example_eval_openai_api.py
```
You will see the final output as follows:

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
