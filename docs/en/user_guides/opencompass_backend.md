# OpenCompass Backend

To facilitate the use of the OpenCompass evaluation backend, we have customized the OpenCompass source code and named it `ms-opencompass`. This version optimizes task configuration and execution and supports installation via PyPI, enabling users to initiate lightweight OpenCompass evaluation tasks through EvalScope. Additionally, we initially provide an interface for evaluation tasks based on the OpenAI API format. You can deploy model services using [ms-swift](https://github.com/modelscope/swift), where [swift deploy](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm) supports using [vLLM](https://github.com/vllm-project/vllm) to launch model inference services.

## 1. Environment Setup
```shell
# Install OpenCompass dependencies
pip install evalscope[opencompass] -U
```

## 2. Data Preparation
````{note}
There are three ways to download datasets. The automatic download method supports fewer datasets than the manual download method. Please use as needed. For detailed dataset information, refer to the [OpenCompass Dataset List](https://hub.opencompass.org.cn/home).
You can view the dataset name list using the following code:
```python
from evalscope.backend.opencompass import OpenCompassBackendManager
print(f'All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')
```
````

`````{tabs}
````{tab} Automatic Download (Recommended)
You can automatically download datasets from ModelScope. To enable this feature, set the following environment variable:
```shell
export DATASET_SOURCE=ModelScope
```
The following datasets will be automatically downloaded during usage:
```text
humaneval, triviaqa, commonsenseqa, tydiqa, strategyqa, cmmlu, lambada, piqa, ceval, math, LCSTS, Xsum, winogrande, openbookqa, AGIEval, gsm8k, nq, race, siqa, mbpp, mmlu, hellaswag, ARC, BBH, xstory_cloze, summedits, GAOKAO-BENCH, OCNLI, cmnli
```
````
````{tab} Using ModelScope
```shell
# Download
wget -O eval_data.zip https://www.modelscope.cn/datasets/swift/evalscope_resource/resolve/master/eval.zip
# Unzip
unzip eval_data.zip
```
Included datasets:
```text
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada', 'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze', 'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval', 'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench', 'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
Total size: approximately 1.7GB. After downloading and unzipping, place the dataset folder (i.e., the data folder) in your current working directory.
````
````{tab} Using GitHub
```shell
# Download
wget -O eval_data.zip https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
# Unzip
unzip eval_data.zip
```
Included datasets:
```text
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada', 'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze', 'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval', 'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench', 'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
Total size: approximately 1.7GB. After downloading and unzipping, place the dataset folder (i.e., the data folder) in your current working directory.
````
`````

## 3. Model Inference Service
OpenCompass evaluation backend uses a unified OpenAI API call for assessment, so we need to deploy the model. 

Here are three ways to deploy model services:

`````{tabs}
````{tab} ms-swift (Recommended)
Use ms-swift to deploy model services. For more details, please refer to the: [ms-swift Deployment Guide](https://swift.readthedocs.io/en/latest/LLM/VLLM-inference-acceleration-and-deployment.html), which natively supports three deployment methods: pt, vllm, and lmdeploy.

**Install ms-swift**
```shell
pip install ms-swift -U
```
**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-0_5b-instruct --port 8000
```
````

````{tab} vLLM
Refer to [vLLM Tutorial](https://docs.vllm.ai/en/latest/index.html) for more details.

**Install vLLM**
```shell
pip install vllm -U
```
**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen2-0.5B-Instruct --port 8000
```
````

````{tab} LMDeploy
Refer to [LMDeploy Tutorial](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/api_server_vl.md) for more details.

**Install LMDeploy**
```shell
pip install lmdeploy -U
```
**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server Qwen2-0.5B-Instruct --server-port 8000
```
````
`````


## 4. Model Evaluation

### Configuration File
There are three ways to write the configuration file:
`````{tabs}
````{tab} Python Dictionary
```python
task_cfg_dict = dict(
    eval_backend='OpenCompass',
    eval_config={
        'datasets': ["mmlu", "ceval", 'ARC_c', 'gsm8k'],
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
````{tab} YAML Configuration File
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
````{tab} JSON Configuration File
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

#### Parameter Descriptions
- `eval_backend`: Default value is `OpenCompass`, indicating the use of the OpenCompass evaluation backend.
- `eval_config`: A dictionary containing the following fields:
  - `datasets`: A list, referring to the [currently supported datasets](#2-data-preparation).
  - `models`: A list of dictionaries, each dictionary must contain the following fields:
    - `path`: Value reused from the `--model_type` in the `swift deploy` command; If using `vLLM` or `LMDeploy` to deploy a model, set it to `model_id`.
    - `openai_api_base`: The URL for the OpenAI API, which is the URL for the Swift model service.
    - `is_chat`: Boolean value, set to `True` indicates a chat model; set to `False` indicates a base model.
    - `key`: The OpenAI API key for the model API, default value is `EMPTY`.
  - `work_dir`: A string specifying the directory to save evaluation results, logs, and summaries. Default value is `outputs/default`.
  - `limit`: Can be `int`, `float`, or `str`, for example, 5, 5.0, or `'[10:20]'`. Default value is `None`, which indicates all examples will be run.
For other optional attributes, refer to `opencompass.cli.arguments.ApiModelConfig`.

### Run Script
After configuring the configuration file, run the following script:
```python
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_swift_eval():
    # Option 1: Python dictionary
    task_cfg = task_cfg_dict

    # Option 2: YAML configuration file
    # task_cfg = 'eval_swift_openai_api.yaml'

    # Option 3: JSON configuration file
    # task_cfg = 'eval_swift_openai_api.json'
    
    run_task(task_cfg=task_cfg)
    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_swift_eval()
```
Or run the following command:
```shell
python examples/example_eval_swift_openai_api.py
```
You will see the final output as follows:

```text
dataset                                 version    metric         mode    qwen2-0_5b-instruct
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