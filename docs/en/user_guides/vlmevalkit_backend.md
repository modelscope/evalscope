# VLMEvalKit Backend

To facilitate the use of the VLMEvalKit evaluation backend, we have customized the VLMEvalKit source code, naming it `ms-vlmeval`. This version encapsulates the configuration and execution of evaluation tasks and supports installation via PyPI, allowing users to initiate lightweight VLMEvalKit evaluation tasks through EvalScope. Additionally, we support interface evaluation tasks based on the OpenAI API format, and you can deploy multi-modal model services using ModelScope [swift](https://github.com/modelscope/swift).

## 1. Environment Setup
```shell
# Install additional dependencies
pip install evalscope[vlmeval]
```

## 2. Data Preparation
The currently supported datasets include:
```text
'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'DocVQA_VAL', 'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision', 'MathVision_MINI', 'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet', 'MTVQA_TEST', 'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL', 'VCR_EN_HARD_500', 'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100', 'VCR_ZH_EASY_ALL', 'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMBench-Video', 'Video-MME', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK'
```
For detailed information about the datasets, refer to the [VLMEvalKit Supported Multimodal Benchmark List](https://swift.readthedocs.io/en/latest/LLM/Supported-models-datasets.html).

````{note}
When loading a dataset, if the local dataset file does not exist, it will be automatically downloaded to the `~/LMUData/` directory.
You can view the dataset name list using the following code:
```python
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_models().keys()}')
```
````

## 3. Model Evaluation
Model evaluation can be conducted in two ways: through deployed model services or local model inference. Details are as follows:

### Method 1: Deployed Model Service Evaluation
#### Model Deployment
Here are four ways to deploy model services:

`````{tabs}
````{tab} ms-swift (Recommended)
Use ms-swift to deploy model services. For more details, please refer to the: [ms-swift Deployment Guide](https://swift.readthedocs.io/en/latest/Multi-Modal/mutlimodal-deployment.html).

**Install ms-swift**
```shell
pip install ms-swift -U
```
**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat --port 8000
```
````

````{tab} vLLM 
Refer to the [vLLM Tutorial](https://docs.vllm.ai/en/latest/index.html) for more details.

[List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models)

**Install vLLM**
```shell
pip install vllm -U
```

**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model InternVL2-8B --port 8000 --trust-remote-code --max_model_len 4096
```
````

````{tab} LMDeploy 
Refer to [LMDeploy Tutorial](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/api_server_vl.md) for more details.

**Install LMDeploy**
```shell
pip install lmdeploy -U
```
**Deploy Model Service**s
```shell
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server Qwen-VL-Chat --server-port 8000
```
````

````{tab} Ollama
```{note}
Support for OpenAI API by Ollama is currently in an experimental state. This tutorial provides an example only; please modify it according to your actual situation.
```

Refer to the [Ollama Tutorial](https://github.com/ollama/ollama/blob/main/README.md#quickstart).

**Install Ollama**

```shell
# For Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Start Ollama**
```shell
# Default port is 11434
ollama serve
```

```{tip}
If using `ollama pull` to fetch a model, you can skip the following steps for creating a model; if using `ollama import` to import a model, you will need to manually create a model configuration file.
```

**Create Model Configuration File `Modelfile`**

[Supported Model Formats](https://github.com/ollama/ollama/blob/main/docs/import.md)

```text
# Model path
FROM models/LLaVA

# Temperature coefficient
PARAMETER temperature 1

# System prompt
SYSTEM """
You are a helpful assistant.
"""
```

**Create Model**

This command will automatically convert the model to a format supported by Ollama, with support for various quantization methods.
```shell
ollama create llava -f ./Modelfile
```
````

`````

#### Configure Model Evaluation Parameters
Create configuration files:
`````{tabs}
````{tab} YAML Configuration File
```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: qwen-vl-chat
      name: CustomAPIModel 
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
  data:
    - SEEDBench_IMG
    - ChartQA_TEST
  mode: all
  limit: 20
  rerun: true
  work_dir: outputs
  nproc: 16
```
````
````{tab} Python Dictionary
```python
task_cfg_dict = {
    'eval_backend': 'VLMEvalKit',
    'eval_config': 
            {'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
            'limit': 20,
            'mode': 'all',
            'model': [ 
                {'api_base': 'http://localhost:8000/v1/chat/completions',
                'key': 'EMPTY',
                'name': 'CustomAPIModel',
                'temperature': 0.0,
                'type': 'qwen-vl-chat'}
                ],
            'rerun': True,
            'work_dir': 'output'}}
```
````
`````
#### Basic Parameters
- `eval_backend`: Default value is `VLMEvalKit`, indicating the use of the VLMEvalKit evaluation backend.
- `eval_config`: A dictionary containing the following fields:
  - `data`: A list referencing the [currently supported datasets](#2-data-preparation).
  - `model`: A list of dictionaries; each must contain the following fields:
    - `type`: The model name for OpenAI API requests.
      - If deploying with `ms-swift`, set to the value of `--model_type`;
      - If deploying with `vLLM` or `LMDeploy`, set to `model_id`;
      - If deploying with `Ollama`, set to `model_name`, and use the `ollama list` command to check.
    - `name`: Fixed value, must be `CustomAPIModel`.
    - `api_base`: The URL for the OpenAI API, which is the URL for the Swift model service.
    - `key`: The OpenAI API key for the model API, default value is `EMPTY`.
    - `temperature`: Temperature coefficient for model inference; default value is `0.0`.
    - `img_size`: Image size for model inference; default value is `-1`, indicating the original size; set to other values, e.g., `224`, to resize the image to 224x224.
  - `mode`: Options: `['all', 'infer']`; `all` includes inference and evaluation; `infer` only performs inference.
  - `limit`: Integer indicating the number of evaluation data; default value is `None`, meaning all examples will be run.
  - `rerun`: Boolean indicating whether to rerun the evaluation, which will delete all temporary evaluation files.
  - `work_dir`: String specifying the directory to save evaluation results, logs, and summaries; default value is `outputs`.
  - `nproc`: Integer indicating the number of API calls in parallel.
For other optional parameters, refer to `vlmeval.utils.arguments`.

### Method 2: Local Model Inference Evaluation
This method does not involve starting a model service; instead, it directly configures model evaluation parameters for local inference.

#### Configure Model Evaluation Parameters
`````{tabs}
````{tab} YAML Configuration File
```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - name: qwen_chat
      model_path: models/Qwen-VL-Chat
  data:
    - SEEDBench_IMG
    - ChartQA_TEST
  mode: all
  limit: 20
  rerun: true
  work_dir: outputs
  nproc: 16
```
````
````{tab} Python Dictionary
```python
task_cfg_dict = {
    'eval_backend': 'VLMEvalKit',
    'eval_config': 
            {'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
            'limit': 20,
            'mode': 'all',
            'model': [ 
                {'name': 'qwen_chat',
                'model_path': 'models/Qwen-VL-Chat'}
                ],
            'rerun': True,
            'work_dir': 'output'}}
```
````
`````
#### Parameter Descriptions
The [basic parameters](#basic-parameters) are consistent with the deployed model service evaluation method, but the model parameters differ:
- `model`: A list of dictionaries where each model requires different fields:
  - `name`: Model name, refer to the [models supported by VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py).
  - `model_path` and other parameters: Refer to the [model parameters supported by VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py).

### (Optional) Deploy Judge Model
You can deploy a local language model as a judge/extractor using ms-swift. Refer to the [ms-swift LLM Deployment Guide](https://swift.readthedocs.io/en/latest/Multi-Modal/vllm-inference-acceleration.html).

````{note}
If the judge model is not deployed, post-processing with exact matching will be used; also, **you must configure the judge model environment variables for correct model invocation**.
````

#### Deploy Judge Model
```shell
# Deploy qwen2-7b as the judge
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct --model_id_or_path models/Qwen2-7B-Instruct --port 8866
```

#### Configure Judge Model Environment Variables
Add the following configuration in the YAML configuration file:
```yaml
OPENAI_API_KEY: EMPTY 
OPENAI_API_BASE: http://127.0.0.1:8866/v1/chat/completions # Judge model's api_base
LOCAL_LLM: qwen2-7b-instruct # Judge model's model_id
```

## 4. Execute Evaluation Task
After configuring the configuration file, run the following script:
```python
from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_swift_eval():
    # Option 1: Python dictionary
    task_cfg = task_cfg_dict
    # Option 2: YAML configuration file
    # task_cfg = 'eval_swift_openai_api.yaml'
    
    run_task(task_cfg=task_cfg)
    print('>> Start to get the report with summarizer ...')
    report_list = Summarizer.get_report_from_cfg(task_cfg)
    print(f'\n>> The report list: {report_list}')

run_swift_eval()
```