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
### Model Deployment

Here are four ways to deploy model services:

::::{tab-set}

:::{tab-item} vLLM Deployment

Refer to the [vLLM Tutorial](https://docs.vllm.ai/en/latest/index.html) for more details.

[List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models)

**Install vLLM**
```shell
pip install vllm -U
```

**Deploy Model Service**
```shell
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-3B-Instruct --port 8000 --trust-remote-code --max_model_len 4096 --served-model-name Qwen2.5-VL-3B-Instruct
```

```{tip}
If you encounter the error `ValueError: At most 1 image(s) may be provided in one request`, try setting the parameter `--limit-mm-per-prompt "image=5"` and you can set the image to a larger value.
```
:::

:::{tab-item} ms-swift Deployment

Deploy model services using ms-swift. For more details, refer to the [ms-swift Deployment Guide](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html#deployment).

**Install ms-swift**
```shell
pip install ms-swift -U
```

**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2.5-VL-3B-Instruct --port 8000
```
:::

:::{tab-item} LMDeploy Deployment

Refer to the [LMDeploy Tutorial](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/api_server_vl.md).

**Install LMDeploy**
```shell
pip install lmdeploy -U
```

**Deploy Model Service**
```shell
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server Qwen-VL-Chat --server-port 8000
```
:::

:::{tab-item} Ollama Deployment

Run ModelScope hosted models with Ollama in one click. Refer to the [documentation](https://www.modelscope.cn/docs/models/advanced-usage/ollama-integration).

```shell
ollama run modelscope.cn/IAILabs/Qwen2.5-VL-7B-Instruct-GGUF
```

:::

::::

#### Configure Model Evaluation Parameters

Write configuration:

::::{tab-set}

:::{tab-item} YAML Configuration File

```yaml
work_dir: outputs
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: Qwen2.5-VL-3B-Instruct
      name: CustomAPIModel 
      api_base: http://localhost:8000/v1/chat/completions
      key: EMPTY
      temperature: 0.0
      img_size: -1
      max_tokens: 1024
      video_llm: false
  data:
    - SEEDBench_IMG
    - ChartQA_TEST
  mode: all
  limit: 20
  reuse: false
  nproc: 16
  judge: exact_matching
```
:::

:::{tab-item} TaskConfig Dictionary

```python
from evalscope import TaskConfig

task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config={
        'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
        'limit': 20,
        'mode': 'all',
        'model': [ 
            {'api_base': 'http://localhost:8000/v1/chat/completions',
            'key': 'EMPTY',
            'name': 'CustomAPIModel',
            'temperature': 0.0,
            'type': 'Qwen2.5-VL-3B-Instruct',
            'img_size': -1,
            'video_llm': False,
            'max_tokens': 1024,}
            ],
        'reuse': False,
        'nproc': 16,
        'judge': 'exact_matching'}
)
```
:::

::::

### Method 2: Local Model Inference Evaluation

Configure model evaluation parameters directly for local inference without starting the model service.

#### Configure Model Evaluation Parameters

::::{tab-set}

:::{tab-item} YAML Configuration File

```{code-block} yaml 
:caption: eval_openai_api.json

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

:::{tab-item} TaskConfig Dictionary

```python
from evalscope import TaskConfig

task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config=
        {'data': ['SEEDBench_IMG', 'ChartQA_TEST'],
        'limit': 20,
        'mode': 'all',
        'model': [ 
            {'name': 'qwen_chat',
            'model_path': 'models/Qwen-VL-Chat',
            'video_llm': False,
            'max_new_tokens': 1024,
            }
          ],
        'reuse': False}
 )
```
:::

::::

### Parameter Explanation

- `eval_backend`: Default value is `VLMEvalKit`, indicating the use of VLMEvalKit as the evaluation backend.
- `work_dir`: String, the directory for saving evaluation results, logs, and summaries. Default value is `outputs`.
- `eval_config`: Dictionary containing the following fields:
  - `data`: List, refer to the [currently supported datasets](#2-data-preparation)
  - `model`: List of dictionaries, each specifying the following fields:
    - For remote API calls:
      - `api_base`: URL of the model service.
      - `type`: API request model name, e.g., `Qwen2.5-VL-3B-Instruct`.
      - `name`: Fixed value, must be `CustomAPIModel`.
      - `key`: OpenAI API key for the model API, default is `EMPTY`.
      - `temperature`: Temperature coefficient for model inference, default is `0.0`.
      - `max_tokens`: Maximum number of tokens for model inference, default is `2048`.
      - `img_size`: Image size for model inference, default is `-1`, meaning use the original size; set to other values, e.g., `224`, to scale the image to 224x224.
      - `video_llm`: Boolean, default is `False`. Set to `True` to pass `video_url` parameter when evaluating video datasets.
    - For local model inference:
      - `name`: Model name, refer to [VLMEvalKit supported models](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py).
      - `model_path` and other parameters: refer to [VLMEvalKit supported model parameters](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py).
  - `mode`: Options: `['all', 'infer']`, `all` includes inference and evaluation; `infer` performs inference only.
  - `limit`: Integer, number of data items to evaluate, default is `None`, meaning run all examples.
  - `reuse`: Boolean, whether to reuse evaluation results, otherwise all evaluation temporary files will be deleted.
    ```{note}
    For `ms-vlmeval>=0.0.11`, the parameter `rerun` is renamed to `reuse`, default is `False`. When set to `True`, you need to add `use_cache` in `task_cfg_dict` to specify the cache directory.
    ```
  - `nproc`: Integer, number of API calls to be made in parallel.
  - `nframe`: Integer, number of video frames for video datasets, default is `8`.
  - `fps`: Integer, frame rate for video datasets, default is `-1`, meaning use `nframe`; set greater than 0 to use `fps` to calculate the number of video frames.
  - `use_subtitle`: Boolean, whether to use subtitles for video datasets, default is `False`.


### (Optional) Deploy Judge Model
Deploy a local language model as a judge / choice extractor, also using ms-swift to deploy the model service. For details, refer to: [ms-swift LLM Deployment Guide](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html).
````{note}
When no judge model is deployed, post-processing + exact matching will be used for judging; and **the judge model environment variables must be configured to correctly call the model**.
````

#### Deploy Judge Model
```shell
# Deploy qwen2-7b as the judge
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct --model_id_or_path models/Qwen2-7B-Instruct --port 8866
```

#### Configure Judge Model Environment Variables
Add the following configuration to `eval_config` in the yaml configuration file:
```yaml
eval_config:
  # ... other configurations
  OPENAI_API_KEY: EMPTY 
  OPENAI_API_BASE: http://127.0.0.1:8866/v1/chat/completions # api_base of the judge model
  LOCAL_LLM: qwen2-7b-instruct # model_id of the judge model
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
