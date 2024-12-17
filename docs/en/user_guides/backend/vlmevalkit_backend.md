(vlmeval)=
# VLMEvalKit

To facilitate the use of the VLMEvalKit evaluation backend, we have customized the VLMEvalKit source code, naming it `ms-vlmeval`. This version encapsulates the configuration and execution of evaluation tasks and supports installation via PyPI, allowing users to initiate lightweight VLMEvalKit evaluation tasks through EvalScope. Additionally, we support interface evaluation tasks based on the OpenAI API format, and you can deploy multi-modal model services using [ms-swift](https://github.com/modelscope/swift) , [vLLM](https://github.com/vllm-project/vllm) , [LMDeploy](https://github.com/InternLM/lmdeploy) , [Ollama](https://ollama.ai/), etc.

## 1. Environment Setup
```shell
# Install additional dependencies
pip install evalscope[vlmeval]
```

## 2. Data Preparation
When loading a dataset, if the local dataset file does not exist, it will be automatically downloaded to the `~/LMUData/` directory.

The currently supported datasets include:

| Name                                                               | Notes                                                      |
|--------------------------------------------------------------------|--------------------------------------------------------------|
| A-Bench_TEST, A-Bench_VAL                                          |                                                              |
| AI2D_TEST, AI2D_TEST_NO_MASK                                       |                                                              |
| AesBench_TEST, AesBench_VAL                                        |                                                              |
| BLINK                                                              |                                                              |
| CCBench                                                            |                                                              |
| COCO_VAL                                                           |                                                              |
| ChartQA_TEST                                                       |                                                              |
| DUDE, DUDE_MINI                                                   |                                                              |
| DocVQA_TEST, DocVQA_VAL                                            | DocVQA_TEST does not provide answers; use DocVQA_VAL for automatic evaluation  |
| GMAI_mm_bench_VAL                                                  |                                                              |
| HallusionBench                                                     |                                                              |
| InfoVQA_TEST, InfoVQA_VAL                                          | InfoVQA_TEST does not provide answers; use InfoVQA_VAL for automatic evaluation  |
| LLaVABench                                                         |                                                              |
| MLLMGuard_DS                                                       |                                                              |
| MMBench-Video                                             |                                                              |
| MMBench_DEV_CN, MMBench_DEV_CN_V11                                 |                                                              |
| MMBench_DEV_EN, MMBench_DEV_EN_V11                                 |                                                              |
| MMBench_TEST_CN, MMBench_TEST_CN_V11                               | MMBench_TEST_CN does not provide answers                     |
| MMBench_TEST_EN, MMBench_TEST_EN_V11                               | MMBench_TEST_EN does not provide answers                     |
| MMBench_dev_ar, MMBench_dev_cn, MMBench_dev_en,                   |                                                              |
| MMBench_dev_pt, MMBench_dev_ru, MMBench_dev_tr                     |                                                              |
| MMDU                                                              |                                                              |
| MME                                                               |                                                              |
| MMLongBench_DOC                                                    |                                                              |
| MMMB, MMMB_ar, MMMB_cn, MMMB_en,                                   |                                                              |
| MMMB_pt, MMMB_ru, MMMB_tr                                          |                                                              |
| MMMU_DEV_VAL, MMMU_TEST                                            |                                                              |
| MMStar                                                            |                                                              |
| MMT-Bench_ALL, MMT-Bench_ALL_MI,                                   |                                                              |
| MMT-Bench_VAL, MMT-Bench_VAL_MI                                    |                                                              |
| MMVet                                                             |                                                              |
| MTL_MMBench_DEV                                                   |                                                              |
| MTVQA_TEST                                                         |                                                              |
| MVBench, MVBench_MP4                                               |                                                              |
| MathVision, MathVision_MINI, MathVista_MINI                       |                                                              |
| OCRBench                                                          |                                                              |
| OCRVQA_TEST, OCRVQA_TESTCORE                                      |                                                              |
| POPE                                                              |                                                              |
| Q-Bench1_TEST, Q-Bench1_VAL                                        |                                                              |
| RealWorldQA                                                       |                                                              |
| SEEDBench2, SEEDBench2_Plus, SEEDBench_IMG                        |                                                              |
| SLIDEVQA, SLIDEVQA_MINI                                           |                                                              |
| ScienceQA_TEST, ScienceQA_VAL                                      |                                                              |
| TaskMeAnything_v1_imageqa_random                                   |                                                              |
| TextVQA_VAL                                                        |                                                              |
| VCR_EN_EASY_100, VCR_EN_EASY_500, VCR_EN_EASY_ALL                |                                                              |
| VCR_EN_HARD_100, VCR_EN_HARD_500, VCR_EN_HARD_ALL                |                                                              |
| VCR_ZH_EASY_100, VCR_ZH_EASY_500, VCR_ZH_EASY_ALL                |                                                              |
| VCR_ZH_HARD_100, VCR_ZH_HARD_500, VCR_ZH_HARD_ALL                |                                                              |
| Video-MME                                                         |                                                              |


````{note}
For detailed information about the datasets, refer to the [VLMEvalKit Supported Multimodal Benchmark List](https://swift.readthedocs.io/en/latest/LLM/Supported-models-datasets.html). 

You can view the dataset name list using the following code:
```python
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_datasets()}')
```
````

## 3. Model Evaluation
Model evaluation can be conducted in two ways: through deployed model services or local model inference. Details are as follows:

### Method 1: Deployed Model Service Evaluation
#### Model Deployment
Here are four ways to deploy model services:

::::{tab-set}
:::{tab-item} ms-swift (Recommended)
Use ms-swift to deploy model services. For more details, please refer to the: [ms-swift Deployment Guide](https://swift.readthedocs.io/en/latest/Multi-Modal/mutlimodal-deployment.html).

**Install ms-swift**
```shell
pip install ms-swift -U
```
**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat --port 8000
```
:::

:::{tab-item} vLLM 
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
:::

:::{tab-item} LMDeploy 
Refer to [LMDeploy Tutorial](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/api_server_vl.md) for more details.

**Install LMDeploy**
```shell
pip install lmdeploy -U
```
**Deploy Model Service**s
```shell
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server Qwen-VL-Chat --server-port 8000
```
:::

:::{tab-item} Ollama
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
:::

::::

#### Configure Model Evaluation Parameters
Create configuration files:
::::{tab-set}
:::{tab-item} YAML Configuration File
```yaml
work_dir: outputs
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
  reuse: false
  nproc: 16
```
:::

:::{tab-item} Python Dictionary
```python
task_cfg_dict = {
    'work_dir': 'outputs',
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
            'reuse': False}}
```
:::
::::

### Method 2: Local Model Inference Evaluation
This method does not involve starting a model service; instead, it directly configures model evaluation parameters for local inference.

#### Configure Model Evaluation Parameters
::::{tab-set}
:::{tab-item} YAML Configuration File
```yaml
work_dir: outputs
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
  reuse: false
  work_dir: outputs
  nproc: 16
```
:::

:::{tab-item} Python Dictionary
```python
task_cfg_dict = {
    'work_dir': 'outputs',
    'eval_backend': 'VLMEvalKit',
    'eval_config': 
            {'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
            'limit': 20,
            'mode': 'all',
            'model': [ 
                {'name': 'qwen_chat',
                'model_path': 'models/Qwen-VL-Chat'}
                ],
            'reuse': False}}
```
:::
::::

## Parameters
- `work_dir`: A string specifying the directory where evaluation results, logs, and summaries are saved. The default value is `outputs`.
- `eval_backend`: Default value is `VLMEvalKit`, indicating the use of the VLMEvalKit evaluation backend.
- `eval_config`: A dictionary containing the following fields:
  - `data`: A list referencing the [currently supported datasets](#2-data-preparation).
Certainly! Here's the translated text in English, while maintaining the original formatting:

  - `model`: A list of dictionaries, where each dictionary can specify the following fields:
    - When using remote API calls:
      - `api_base`: The URL for the OpenAI API, i.e., the URL of the model service.
      - `type`: The name of the model for OpenAI API requests.
        - If deploying with `ms-swift`, set to the value of `--model_type`;
        - If deploying the model with `vLLM` or `LMDeploy`, set to `model_id`;
        - If deploying the model with `Ollama`, set to `model_name`, you can check it using the `ollama list` command.
      - `name`: Fixed value, must be `CustomAPIModel`.
      - `key`: The OpenAI API key for the model API, default is `EMPTY`.
      - `temperature`: The temperature coefficient for model inference, default is `0.0`.
      - `img_size`: The image size for model inference, default is `-1`, which means using the original size; setting it to another value, such as `224`, means scaling the image to 224x224 size.
      - `video_llm`: Boolean value, default is `False`. When evaluating video datasets, if you need to pass the `video_url` parameter, set it to `True`.
    - When using local model inference:
      - `name`: Model name, refer to [Models Supported by VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py).
      - `model_path` and other parameters: Refer to [VLMEvalKit Supported Model Parameters](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py).
  - `mode`: Options: `['all', 'infer']`; `all` includes inference and evaluation; `infer` only performs inference.
  - `limit`: Integer indicating the number of evaluation data; default value is `None`, meaning all examples will be run.
  - `reuse`: Boolean indicating whether to reuse the evaluation, which will delete all temporary evaluation files.
    ```{note}
    For `ms-vlmeval>=0.0.11`, the parameter `rerun` has been renamed to `reuse`, with a default value of `False`. When set to `True`, you need to add `use_cache` in the `task_cfg_dict` to specify the cache directory to be used.
    ```
  - `work_dir`: String specifying the directory to save evaluation results, logs, and summaries; default value is `outputs`.
  - `nproc`: Integer indicating the number of API calls in parallel.
  - `nframe`: An integer representing the number of video frames in the video dataset, with a default value of `8`.
  - `fps`: An integer representing the frame rate of the video dataset, with a default value of `-1`, which means to use `nframe`; if set to a value greater than 0, it will use `fps` to calculate the number of video frames.
  - `use_subtitle`: A boolean value indicating whether the video dataset uses subtitles, with a default value of `False`.



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

```{caution}
If you want the model to perform inference again, you need to clear the model prediction results in the `outputs` folder before running the script. Previous prediction results will not be automatically cleared, and **if they exist, the inference phase will be skipped** and the results will be evaluated directly.
```

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
