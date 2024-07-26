[English](README.md) | 简体中文

<p align="center">
<a href="https://pypi.org/project/llmuses"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/llmuses">
</a>
<a href="https://github.com/modelscope/eval-scope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
<p>

## 📖 目录
- [简介](#简介)
- [新闻](#新闻)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [数据集列表](#数据集列表)
- [Leaderboard榜单](#leaderboard-榜单)
- [实验和报告](#实验和报告)
- [性能评测工具](#性能评测工具)


## 📝 简介
大型语言模型评估（LLMs evaluation）已成为评价和改进大模型的重要流程和手段，为了更好地支持大模型的评测，我们提出了Eval-Scope框架，该框架主要包括以下几个部分：
- 预置了多个常用的测试基准数据集，包括：MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH、HumanEval等
- 常用评估指标（metrics）的实现
- 统一model接入，兼容多个系列模型的generate、chat接口
- 自动评估（evaluator）：
    - 客观题自动评估
    - 使用专家模型实现复杂任务的自动评估
- 评估报告生成
- 竞技场模式(Arena)
- 可视化工具
- [模型性能评估](llmuses/perf/README.md)
- 支持OpenCompass作为评测后段，对其进行了高级封装和任务简化，您可以更轻松地提交任务到OpenCompass进行评估。
- 支持VLMEvalKit作为评测后端，通过Eval-Scope作为入口，发起VLMEvalKit的多模态评测任务，支持多种多模态模型和数据集。
- 全链路支持：通过与SWIFT的无缝集成，您可以轻松地训练和部署模型服务，发起评测任务，查看评测报告，实现一站式大模型开发流程。

**特点**
- 轻量化，尽量减少不必要的抽象和配置
- 易于定制
  - 仅需实现一个类即可接入新的数据集
  - 模型可托管在[ModelScope](https://modelscope.cn)上，仅需model id即可一键发起评测
  - 支持本地模型可部署在本地
  - 评估报告可视化展现
- 丰富的评估指标
- model-based自动评估流程，支持多种评估模式
  - Single mode: 专家模型对单个模型打分
  - Pairwise-baseline mode: 与 baseline 模型对比
  - Pairwise (all) mode: 全部模型两两对比


## 🎉 新闻
- **[2024.07.26]:** 支持**VLMEvalKit**作为第三方评测框架，发起多模态模型评测任务，[使用指南](#vlmevalkit-评测后端) 🔥🔥🔥
- **[2024.06.29]:** 支持**OpenCompass**作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评估任务配置，[使用指南](#opencompass-评测后端) 🔥🔥🔥
- **[2024.06.13]:** Eval-Scope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 🚀🚀🚀
- **[2024.06.13]:** 接入Agent评测集ToolBench 🚀🚀🚀



## 🛠️ 环境准备
### 使用pip安装
我们推荐使用conda来管理环境，并使用pip安装依赖:
1. 创建conda环境
```shell
conda create -n eval-scope python=3.10
conda activate eval-scope
```
2. 安装依赖
```shell
pip install llmuses
```

### 使用源码安装
1. 下载源码
```shell
git clone https://github.com/modelscope/eval-scope.git
```
2. 安装依赖
```shell
cd eval-scope/
pip install -e .
```


## 🚀 快速开始

### 简单评估
在指定的若干数据集上评估某个模型，流程如下：
如果使用git安装，可在任意路径下执行：
```shell
python -m llmuses.run --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --limit 100
```
如果使用源码安装，在eval-scope路径下执行：
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets mmlu ceval --limit 10
```
其中，--model参数指定了模型的ModelScope model id，模型链接：[ZhipuAI/chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary)

### 带参数评估
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --model-args revision=v1.0.2,precision=torch.float16,device_map=auto --datasets mmlu ceval --use-cache true --limit 10
```
```
python llmuses/run.py --model qwen/Qwen-1_8B --generation-config do_sample=false,temperature=0.0 --datasets ceval --dataset-args '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}' --limit 10
```
参数说明：
- --model-args: 模型参数，以逗号分隔，key=value形式
- --datasets: 数据集名称，支持输入多个数据集，使用空格分开，参考下文`数据集列表`章节
- --use-cache: 是否使用本地缓存，默认为`false`;如果为`true`，则已经评估过的模型和数据集组合将不会再次评估，直接从本地缓存读取
- --dataset-args: 数据集的evaluation settings，以json格式传入，key为数据集名称，value为参数，注意需要跟--datasets参数中的值一一对应
  - --few_shot_num: few-shot的数量
  - --few_shot_random: 是否随机采样few-shot数据，如果不设置，则默认为true
- --limit: 每个subset最大评估数据量
- --template-type: 需要手动指定该参数，使得eval-scope能够正确识别模型的类型，用来设置model generation config。  

关于--template-type，具体可参考：[模型类型列表](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md)
在模型列表中的`Default Template`字段中找到合适的template；  
可以使用以下方式，来查看模型的template type list：
```shell
from llmuses.models.template import TemplateType
print(TemplateType.get_template_name_list())
```

### 使用评测后端 (Evaluation Backend)
Eval-Scope支持使用第三方评测框架发起评测任务，我们称之为评测后端 (Evaluation Backend)。目前支持的Evaluation Backend有：
- **Native**：Eval-Scope自身的**默认评测框架**，支持多种评估模式，包括单模型评估、竞技场模式、Baseline模型对比模式等。
- [OpenCompass](https://github.com/open-compass/opencompass)：通过Eval-Scope作为入口，发起OpenCompass的评测任务，轻量级、易于定制、支持与LLM微调框架[ModelScope Swift](https://github.com/modelscope/swift)的无缝集成。
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)：通过Eval-Scope作为入口，发起VLMEvalKit的多模态评测任务，支持多种多模态模型和数据集，支持与LLM微调框架[ModelScope Swift](https://github.com/modelscope/swift)的无缝集成。
- **ThirdParty**: 第三方评估任务，如[ToolBench](llmuses/thirdparty/toolbench/README.md)。

#### OpenCompass 评测后端

为便于使用OpenCompass 评测后端，我们基于OpenCompass源码做了定制，命名为`ms-opencompass`，该版本在原版基础上对评估任务的配置和执行做了一些优化，并支持pypi安装方式，使得用户可以通过Eval-Scope发起轻量化的OpenCompass评估任务。同时，我们先期开放了基于OpenAI API格式的接口评估任务，您可以使用[ModelScope Swift](https://github.com/modelscope/swift) 部署模型服务，其中，[swift deploy](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm)支持使用vLLM拉起模型推理服务。

##### 安装
```shell
# 安装额外选项
pip install llmuses[opencompass]
```

##### 数据准备
目前支持的数据集有：
```python
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada', 'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze', 'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval', 'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench', 'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
数据集的详细信息可以参考[OpenCompass数据集列表](https://hub.opencompass.org.cn/home)
您可以使用以下方式，来查看数据集的名称列表：
```python
from llmuses.backend.opencompass import OpenCompassBackendManager
print(f'** All datasets from OpenCompass backend: {OpenCompassBackendManager.list_datasets()}')
```

数据集下载方式：
- 方式1：使用ModelScope数据集下载
    ```shell
    git clone https://www.modelscope.cn/datasets/swift/evalscope_resource.git
    ```

- 方式2：使用github链接下载
    ```shell
    wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
    ```
总大小约1.7GB，下载并解压后，将数据集文件夹（即data文件夹）放置在当前工作路径下。后续我们也即将支持托管在ModelScope上的数据集按需加载方式。


##### 模型推理服务
我们使用ModelScope swift部署模型服务，具体可参考：[ModelScope Swift部署指南](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html#vllm)
```shell
# 安装ms-swift
pip install ms-swift

# 部署模型服务
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type llama3-8b-instruct --port 8000
```


##### 模型评估

参考示例文件： [example_eval_swift_openai_api](examples/example_eval_swift_openai_api.py) 来配置评估任务
执行评估任务：
```shell
python examples/example_eval_swift_openai_api.py
```


#### VLMEvalKit 评测后端

为便于使用VLMEvalKit 评测后端，我们基于VLMEvalKit源码做了定制，命名为`ms-vlmeval`，该版本在原版基础上对评估任务的配置和执行进行了封装，并支持pypi安装方式，使得用户可以通过Eval-Scope发起轻量化的VLMEvalKit评估任务。同时，我们支持基于OpenAI API格式的接口评估任务，您可以使用ModelScope [swift](https://github.com/modelscope/swift) 部署多模态模型服务。

##### 安装
```shell
# 安装额外选项
pip install llmuses[vlmeval]
```

##### 数据准备
目前支持的数据集有：
```python
'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'DocVQA_VAL', 'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision', 'MathVision_MINI', 'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet', 'MTVQA_TEST', 'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL', 'VCR_EN_HARD_500', 'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100', 'VCR_ZH_EASY_ALL', 'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMBench-Video', 'Video-MME', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK'
```
数据集的详细信息可以参考[VLMEvalKit支持的图文多模态评测集](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/README_zh-CN.md#%E6%94%AF%E6%8C%81%E7%9A%84%E5%9B%BE%E6%96%87%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%84%E6%B5%8B%E9%9B%86)
您可以使用以下方式，来查看数据集的名称列表：
```python
from llmuses.backend.vlm_eval_kit import VLMEvalKitBackendManager
print(f'** All models from VLMEvalKit backend: {VLMEvalKitBackendManager.list(list_supported_VLMs().keys())}')
```

在加载数据集时，若本地不存在该数据集文件，将会自动下载数据集到 `~/LMUData/` 目录下。


##### 模型评估
模型评估有两种方式可以选择：

###### 1. ModelScope Swift部署模型服务评估

**模型部署**
使用ModelScope swift部署模型服务，具体可参考：[ModelScope Swift MLLM 部署指南](https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/MLLM%E9%83%A8%E7%BD%B2%E6%96%87%E6%A1%A3.html)
```shell
# 安装ms-swift
pip install ms-swift

# 部署qwen-vl-chat多模态模型服务
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat --model_id_or_path models/Qwen-VL-Chat
```

**模型评估**

参考示例文件： [example_eval_vlm_swift](examples/example_eval_vlm_swift.py) 来配置评估任务
执行评估任务：
```shell
python examples/example_eval_vlm_swift.py
```

###### 2. 本地模型推理评估

**模型推理评估**
不启动模型服务，直接在本地进行推理，参考示例文件： [example_eval_vlm_local](examples/example_eval_vlm_local.py) 来配置评估任务
执行评估任务：
```shell
python examples/example_eval_vlm_local.py
```


##### (可选) 部署裁判员模型
部署本地语言模型作为评判 / 选择提取器，同样使用ModelScope swift部署模型服务，具体可参考：[ModelScope Swift LLM 部署指南](https://swift.readthedocs.io/zh-cn/latest/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.html)
。在未部署裁判员模型模型时，将使用精确匹配。
```shell
# 部署qwen2-7b作为裁判员
CUDA_VISIBLE_DEVICES=1 swift deploy --model_type qwen2-7b-instruct --model_id_or_path models/Qwen2-7B-Instruct --port 8866
```
**必须配置裁判员模型环境变量才能正确调用模型**，需要配置的环境变量如下：
```
OPENAI_API_KEY=EMPTY
OPENAI_API_BASE=http://127.0.0.1:8866/v1/chat/completions # 裁判员模型的api_base
LOCAL_LLM=qwen2-7b-instruct #裁判员模型的 model_id
```


### 使用本地数据集
数据集默认托管在[ModelScope](https://modelscope.cn/datasets)上，加载需要联网。如果是无网络环境，可以使用本地数据集，流程如下：
#### 1. 下载数据集到本地
```shell
# 假如当前本地工作路径为 /path/to/workdir
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
```
则解压后的数据集路径为：/path/to/workdir/data 目录下，该目录在后续步骤将会作为--dataset-dir参数的值传入

#### 2. 使用本地数据集创建评估任务
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --dataset-hub Local --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' --limit 10

# 参数说明
# --dataset-hub: 数据集来源，枚举值： `ModelScope`, `Local`, `HuggingFace` (TO-DO)  默认为`ModelScope`
# --dataset-dir: 当--dataset-hub为`Local`时，该参数指本地数据集路径; 如果--dataset-hub 设置为`ModelScope` or `HuggingFace`，则该参数的含义是数据集缓存路径。
```

#### 3. (可选)在离线环境加载模型和评测
模型文件托管在ModelScope Hub端，需要联网加载，当需要在离线环境创建评估任务时，可参考以下步骤：
```shell
# 1. 准备模型本地文件夹，文件夹结构参考chatglm3-6b，链接：https://modelscope.cn/models/ZhipuAI/chatglm3-6b/files
# 例如，将模型文件夹整体下载到本地路径 /path/to/ZhipuAI/chatglm3-6b

# 2. 执行离线评估任务
python llmuses/run.py --model /path/to/ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --dataset-hub Local --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' --limit 10
```


### 使用run_task函数提交评估任务

#### 1. 配置任务
```python
import torch
from llmuses.constants import DEFAULT_ROOT_CACHE_DIR

# 示例
your_task_cfg = {
        'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
        'generation_config': {'do_sample': False, 'repetition_penalty': 1.0, 'max_new_tokens': 512},
        'dataset_args': {},
        'dry_run': False,
        'model': 'ZhipuAI/chatglm3-6b',
        'template_type': 'chatglm3', 
        'datasets': ['arc', 'hellaswag'],
        'work_dir': DEFAULT_ROOT_CACHE_DIR,
        'outputs': DEFAULT_ROOT_CACHE_DIR,
        'mem_cache': False,
        'dataset_hub': 'ModelScope',
        'dataset_dir': DEFAULT_ROOT_CACHE_DIR,
        'stage': 'all',
        'limit': 10,
        'debug': False
    }

```

#### 2. 执行任务
```python
from llmuses.run import run_task

run_task(task_cfg=your_task_cfg)
```


### 竞技场模式（Arena）
竞技场模式允许多个候选模型通过两两对比(pairwise battle)的方式进行评估，并可以选择借助AI Enhanced Auto-Reviewer（AAR）自动评估流程或者人工评估的方式，最终得到评估报告，流程示例如下：
#### 1. 环境准备
```text
a. 数据准备，questions data格式参考：llmuses/registry/data/question.jsonl
b. 如果需要使用自动评估流程（AAR），则需要配置相关环境变量，我们以GPT-4 based auto-reviewer流程为例，需要配置以下环境变量：
> export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

#### 2. 配置文件
```text
arena评估流程的配置文件参考： llmuses/registry/config/cfg_arena.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    elo_rating: ELO rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```

#### 3. 执行脚本
```shell
#Usage:
cd llmuses

# dry-run模式 (模型answer正常生成，但专家模型，如GPT-4，不会被调用，评估结果会随机生成)
python llmuses/run_arena.py -c registry/config/cfg_arena.yaml --dry-run

# 执行评估流程
python llmuses/run_arena.py --c registry/config/cfg_arena.yaml
```

#### 4. 结果可视化

```shell
# Usage:
streamlit run viz.py -- --review-file llmuses/registry/data/qa_browser/battle.jsonl --category-file llmuses/registry/data/qa_browser/category_mapping.yaml
```


### 单模型打分模式（Single mode）

这个模式下，我们只对单个模型输出做打分，不做两两对比。
#### 1. 配置文件
```text
评估流程的配置文件参考： llmuses/registry/config/cfg_single.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```
#### 2. 执行脚本
```shell
#Example:
python llmuses/run_arena.py --c registry/config/cfg_single.yaml
```

### Baseline模型对比模式（Pairwise-baseline mode）

这个模式下，我们选定 baseline 模型，其他模型与 baseline 模型做对比评分。这个模式可以方便的把新模型加入到 Leaderboard 中（只需要对新模型跟 baseline 模型跑一遍打分即可）
#### 1. 配置文件
```text
评估流程的配置文件参考： llmuses/registry/config/cfg_pairwise_baseline.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```
#### 2. 执行脚本
```shell
# Example:
python llmuses/run_arena.py --c registry/config/cfg_pairwise_baseline.yaml
```


## 数据集列表

| DatasetName        | Link                                                                                   | Status | Note |
|--------------------|----------------------------------------------------------------------------------------|--------|------|
| `mmlu`             | [mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)                         | Active |      |
| `ceval`            | [ceval](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)                  | Active |      |
| `gsm8k`            | [gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)                       | Active |      |
| `arc`              | [arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)                       | Active |      |
| `hellaswag`        | [hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)               | Active |      |
| `truthful_qa`      | [truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)           | Active |      |
| `competition_math` | [competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary) | Active |      |
| `humaneval`        | [humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)               | Active |      |
| `bbh`              | [bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)                           | Active |      |
| `race`             | [race](https://modelscope.cn/datasets/modelscope/race/summary)                         | Active |      |
| `trivia_qa`        | [trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)               | To be intergrated |      |


## Leaderboard 榜单
ModelScope LLM Leaderboard大模型评测榜单旨在提供一个客观、全面的评估标准和平台，帮助研究人员和开发者了解和比较ModelScope上的模型在各种任务上的性能表现。

[Leaderboard](https://modelscope.cn/leaderboard/58/ranking?type=free)



## 实验和报告
参考： [Experiments](./resources/experiments.md)

## 性能评测工具
参考： [性能测试](llmuses/perf/README.md)

## TO-DO List
- [ ] Agents evaluation
- [ ] vLLM
- [ ] Distributed evaluating
- [ ] Multi-modal evaluation
- [ ] Benchmarks
  - [ ] GAIA
  - [ ] GPQA
  - [ ] MBPP
- [ ] Auto-reviewer
  - [ ] Qwen-max


