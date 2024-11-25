(vlmeval)=

# VLMEvalKit

为便于使用VLMEvalKit 评测后端，我们基于VLMEvalKit源码做了定制，命名为`ms-vlmeval`，该版本在原版基础上对评估任务的配置和执行进行了封装，并支持pypi安装方式，使得用户可以通过EvalScope发起轻量化的VLMEvalKit评估任务。同时，我们支持基于OpenAI API格式的接口评估任务，您可以使用[ms-swift](https://github.com/modelscope/swift)、[vLLM](https://github.com/vllm-project/vllm)、[LMDeploy](https://github.com/InternLM/lmdeploy)、[Ollama](https://ollama.ai/)等模型服务，部署多模态模型服务。

## 1. 环境准备
```shell
# 安装额外依赖
pip install evalscope[vlmeval] -U
```

## 2. 数据准备
在加载数据集时，若本地不存在该数据集文件，将会自动下载数据集到 `~/LMUData/` 目录下。

目前支持的数据集有：

| 名称                                                               | 备注   |
|--------------------------------------------------------------------|--------|
| A-Bench_TEST, A-Bench_VAL                                          |        |
| AI2D_TEST, AI2D_TEST_NO_MASK                                       |        |
| AesBench_TEST, AesBench_VAL                                        |        |
| BLINK                                                              |        |
| CCBench                                                            |        |
| COCO_VAL                                                           |        |
| ChartQA_TEST                                                       |        |
| DUDE, DUDE_MINI                                                   |        |
| DocVQA_TEST, DocVQA_VAL                                            |DocVQA_TEST没有提供答案，使用DocVQA_VAL进行自动评估        |
| GMAI_mm_bench_VAL                                                  |        |
| HallusionBench                                                     |        |
| InfoVQA_TEST, InfoVQA_VAL                                          |InfoVQA_TEST没有提供答案，使用InfoVQA_VAL进行自动评估        |
| LLaVABench                                                         |        |
| MLLMGuard_DS                                                       |        |
|MMBench-Video                                             |        |
| MMBench_DEV_CN, MMBench_DEV_CN_V11                                 |        |
| MMBench_DEV_EN, MMBench_DEV_EN_V11                                 |        |
| MMBench_TEST_CN, MMBench_TEST_CN_V11                               | MMBench_TEST_CN没有提供答案       |
| MMBench_TEST_EN, MMBench_TEST_EN_V11                               | MMBench_TEST_EN没有提供答案       |
| MMBench_dev_ar, MMBench_dev_cn, MMBench_dev_en,                   |        |
| MMBench_dev_pt, MMBench_dev_ru, MMBench_dev_tr                     |        |
| MMDU                                                              |        |
| MME                                                               |        |
| MMLongBench_DOC                                                    |        |
| MMMB, MMMB_ar, MMMB_cn, MMMB_en,                                   |        |
| MMMB_pt, MMMB_ru, MMMB_tr                                          |        |
| MMMU_DEV_VAL, MMMU_TEST                                            |        |
| MMStar                                                            |        |
| MMT-Bench_ALL, MMT-Bench_ALL_MI,                                   |        |
| MMT-Bench_VAL, MMT-Bench_VAL_MI                                    |        |
| MMVet                                                             |        |
| MTL_MMBench_DEV                                                   |        |
| MTVQA_TEST                                                         |        |
| MVBench, MVBench_MP4                                               |        |
| MathVision, MathVision_MINI, MathVista_MINI                       |        |
| OCRBench                                                          |        |
| OCRVQA_TEST, OCRVQA_TESTCORE                                      |        |
| POPE                                                              |        |
| Q-Bench1_TEST, Q-Bench1_VAL                                        |        |
| RealWorldQA                                                       |        |
| SEEDBench2, SEEDBench2_Plus, SEEDBench_IMG                        |        |
| SLIDEVQA, SLIDEVQA_MINI                                           |        |
| ScienceQA_TEST, ScienceQA_VAL                                      |        |
| TaskMeAnything_v1_imageqa_random                                   |        |
| TextVQA_VAL                                                        |        |
| VCR_EN_EASY_100, VCR_EN_EASY_500, VCR_EN_EASY_ALL                |        |
| VCR_EN_HARD_100, VCR_EN_HARD_500, VCR_EN_HARD_ALL                |        |
| VCR_ZH_EASY_100, VCR_ZH_EASY_500, VCR_ZH_EASY_ALL                |        |
| VCR_ZH_HARD_100, VCR_ZH_HARD_500, VCR_ZH_HARD_ALL                |        |
| Video-MME                                                         |        |



````{note}
数据集的详细信息可以参考[VLMEvalKit支持的图文多模态评测集](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/README_zh-CN.md#%E6%94%AF%E6%8C%81%E7%9A%84%E5%9B%BE%E6%96%87%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%84%E6%B5%8B%E9%9B%86)。

您可以使用以下方式，来查看数据集的名称列表：
```python
from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list_supported_datasets()}')

```
````



## 3. 模型评估
模型评估有两种方式可以选择，一种是部署模型服务评估，另一种是本地模型推理评估。具体如下：

### 方式1. 部署模型服务评估

#### 模型部署

下面介绍四种方式部署模型服务：
::::{tab-set}
:::{tab-item} ms-swift部署 （推荐）

使用ms-swift部署模型服务，具体可参考：[ms-swift部署指南](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/MLLM%E9%83%A8%E7%BD%B2%E6%96%87%E6%A1%A3.html)。

**安装ms-swift**
```shell
pip install ms-swift -U
```

**部署模型服务**
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat --port 8000
```
:::

:::{tab-item} vLLM 部署

参考[vLLM 教程](https://docs.vllm.ai/en/latest/index.html) for more details.

[支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html#multimodal-language-models)

**安装vLLM**
```shell
pip install vllm -U
```

**部署模型服务**
```shell
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model InternVL2-8B --port 8000 --trust-remote-code --max_model_len 4096
```
:::

:::{tab-item} LMDeploy 部署
参考 [LMDeploy 教程](https://github.com/InternLM/lmdeploy/blob/main/docs/en/multi_modal/api_server_vl.md).

**安装LMDeploy**
```shell
pip install lmdeploy -U
```

**部署模型服务**
```shell
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server Qwen-VL-Chat --server-port 8000
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
FROM models/LLaVA

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
ollama create llava -f ./Modelfile
```
:::
::::


#### 配置模型评估参数

编写配置
::::{tab-set}
:::{tab-item} yaml 配置文件
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
  reuse: true
  work_dir: outputs
  nproc: 16
```
:::

:::{tab-item} Python 字典

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
            'reuse': True,
            'work_dir': 'output'}}
```
:::
::::

#### 基本参数

- `eval_backend`：默认值为 `VLMEvalKit`，表示使用 VLMEvalKit 评测后端。
- `eval_config`：字典，包含以下字段：
  - `data`：列表，参考[目前支持的数据集](#2-数据准备)
  - `model`：字典列表，每个字典必须包含以下字段：
    - `type`：OpenAI API 请求模型名称。
      - 若使用`ms-swift`部署，设置为 `--model_type` 的值；
      - 若使用 `vLLM` 或 `LMDeploy` 部署模型，则设置为 `model_id`；
      - 若使用 `Ollama` 部署模型，则设置为 `model_name`，使用`ollama list`命令查看。
    - `name`：固定值，必须为 `CustomAPIModel`。
    - `api_base`：OpenAI API 的URL，即模型服务的 URL。
    - `key`：模型 API 的 OpenAI API 密钥，默认值为 `EMPTY`。
    - `temperature`：模型推理的温度系数，默认值为 `0.0`。
    - `img_size`：模型推理的图像大小，默认值为 `-1`，表示使用原始大小；设置为其他值，例如 `224`，表示将图像缩放到 224x224 大小。
    - `video_llm`：布尔值，默认为`False`，在评估视频数据集时，如需传递 `video_url` 参数，请设置为 `True`。
  - `mode`：选项: `['all', 'infer']`，`all`包括推理和评估；`infer`仅进行推理。
  - `limit`：整数，评估的数据数量，默认值为 `None`，表示运行所有示例。
  - `reuse`：布尔值，是否重用评估结果，否则将删除所有评估临时文件。
  - `work_dir`：字符串，保存评估结果、日志和摘要的目录。默认值为 `outputs`
  - `nproc`：整数，并行调用 API 的数量。

其他可选参数请参考`vlmeval.utils.arguments`

### 方式2. 本地模型推理评估

不启动模型服务，直接配置模型评估参数，在本地进行推理

#### 配置模型评估参数
::::{tab-set}
:::{tab-item} yaml 配置文件

```{code-block} yaml 
:caption: eval_openai_api.json

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
  reuse: true
  work_dir: outputs
  nproc: 16
```
:::

:::{tab-item} Python 字典

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
            'reuse': True,
            'work_dir': 'outputs'}}
```
:::
::::

#### 参数说明

[基本参数](#基本参数)都与上面部署模型服务评估方式一致，不一样的是模型参数：
- `model`：字典列表，每种模型需要的字段不同
  - `name`：模型名称，参考[VLMEvalKit支持的模型](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py)。
  - `model_path`等其余参数：参考[VLMEvalKit支持的模型参数](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/config.py)。


### (可选) 部署裁判员模型
部署本地语言模型作为评判 / 选择提取器，同样使用ms-swift部署模型服务，具体可参考：[ms-swift LLM 部署指南](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html)
。
````{note}
在未部署裁判员模型模型时，将使用后处理+精确匹配进行评判；且**必须配置裁判员模型环境变量才能正确调用模型**。
````

#### 部署裁判员模型
```shell
# 部署qwen2-7b作为裁判员
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct --model_id_or_path models/Qwen2-7B-Instruct --port 8866
```

#### 配置裁判员模型环境变量
在yaml配置文件中增加如下配置：
```yaml
OPENAI_API_KEY: EMPTY 
OPENAI_API_BASE: http://127.0.0.1:8866/v1/chat/completions # 裁判员模型的 api_base
LOCAL_LLM: qwen2-7b-instruct #裁判员模型的 model_id
```

## 4. 执行评估任务
```{caution}
若想让模型重新进行推理，需清空`outputs`文件夹下的模型预测结果再运行脚本。
因为之前的预测结果不会自动清除，若存在该结果**会跳过推理阶段**，直接对结果进行评测。
```

配置好配置文件后，运行以下脚本即可

```{code-block} python
:caption: eval_openai_api.py

from evalscope.run import run_task
from evalscope.summarizer import Summarizer

def run_eval():
    # 选项 1: python 字典
    task_cfg = task_cfg_dict

    # 选项 2: yaml 配置文件
    # task_cfg = 'eval_openai_api.yaml'

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
