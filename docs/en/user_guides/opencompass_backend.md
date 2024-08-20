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
The OpenCompass evaluation backend uses a unified OpenAI API call for assessments, so model deployment is necessary. Below, we use ms-swift to deploy the model service, further details can be found in the [ms-swift deployment guide](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm).

### Install ms-swift
```shell
pip install ms-swift -U
```

### Deploy Model Service
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-0_5b-instruct --port 8000
# Successful startup log
# INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

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
    - `path`: Value reused from the `--model_type` in the `swift deploy` command.
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