
![](docs/en/_static/images/evalscope_logo.png)

<p align="center">
    <a href="README.md">English</a> | 简体中文
</p>

<p align="center">
  <a href="https://badge.fury.io/py/evalscope"><img src="https://badge.fury.io/py/evalscope.svg" alt="PyPI version" height="18"></a>
  <a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/evalscope">
  </a>
  <a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
  <a href='https://evalscope.readthedocs.io/zh-cn/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/evalscope/badge/?version=latest' alt='Documentation Status' />
  </a>
  <br>
<a href="https://evalscope.readthedocs.io/zh-cn/latest/"> 📖  中文文档</a>
<p>


> ⭐ 如果你喜欢这个项目，请点击右上角的 "Star" 按钮支持我们。你的支持是我们前进的动力！

## 📋 目录
- [简介](#简介)
- [新闻](#新闻)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [使用其他评测后端](#使用其他评测后端)
- [自定义数据集评测](#自定义数据集评测)
- [离线环境评测](#离线环境评测)
- [竞技场模式](#竞技场模式)
- [性能评测工具](#推理性能评测工具)



## 📝 简介

EvalScope是[魔搭社区](https://modelscope.cn/)官方推出的模型评估与性能基准测试框架，内置多个常用测试基准和评估指标，如MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH和HumanEval等；支持多种类型的模型评测，包括LLM、多模态LLM、embedding模型和reranker模型。EvalScope还适用于多种评测场景，如端到端RAG评测、竞技场模式和模型推理性能压测等。此外，通过ms-swift训练框架的无缝集成，可一键发起评测，实现了模型训练到评测的全链路支持🚀

<p align="center">
    <img src="docs/en/_static/images/evalscope_framework.png" style="width: 70%;">
    <br>EvalScope 整体架构图.
</p>

EvalScope包括以下模块：

1. **Model Adapter**: 模型适配器，用于将特定模型的输出转换为框架所需的格式，支持API调用的模型和本地运行的模型。

2. **Data Adapter**: 数据适配器，负责转换和处理输入数据，以便适应不同的评估需求和格式。

3. **Evaluation Backend**:
    - **Native**：EvalScope自身的**默认评测框架**，支持多种评估模式，包括单模型评估、竞技场模式、Baseline模型对比模式等。
    - **OpenCompass**：支持[OpenCompass](https://github.com/open-compass/opencompass)作为评测后端，对其进行了高级封装和任务简化，您可以更轻松地提交任务进行评估。
    - **VLMEvalKit**：支持[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)作为评测后端，轻松发起多模态评测任务，支持多种多模态模型和数据集。
    - **RAGEval**：支持RAG评估，支持使用[MTEB/CMTEB](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html)进行embedding模型和reranker的独立评测，以及使用[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)进行端到端评测。
    - **ThirdParty**：其他第三方评估任务，如ToolBench。

4. **Performance Evaluator**: 模型性能评测，负责具体衡量模型推理服务性能，包括性能评测、压力测试、性能评测报告生成、可视化。

5. **Evaluation Report**: 最终生成的评估报告，总结模型的性能表现，报告可以用于决策和进一步的模型优化。

6. **Visualization**: 可视化结果，帮助用户更直观地理解评估结果，便于分析和比较不同模型的表现。


## 🎉 新闻
- 🔥 **[2024.11.26]** 模型推理压测工具重构完成：支持本地启动推理服务、支持Speed Benchmark；优化异步调用错误处理，参考[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test.html)
- 🔥 **[2024.10.31]** 多模态RAG评测最佳实践发布，参考[📖博客](https://evalscope.readthedocs.io/zh-cn/latest/blog/RAG/multimodal_RAG.html#multimodal-rag)
- 🔥 **[2024.10.23]** 支持多模态RAG评测，包括[CLIP_Benchmark](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/clip_benchmark.html)评估图文检索器，以及扩展了[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)以支持端到端多模态指标评估。
- 🔥 **[2024.10.8]** 支持RAG评测，包括使用[MTEB/CMTEB](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html)进行embedding模型和reranker的独立评测，以及使用[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)进行端到端评测。
- 🔥 **[2024.09.18]** 我们的文档增加了博客模块，包含一些评测相关的技术调研和分享，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/blog/index.html)
- 🔥 **[2024.09.12]** 支持 LongWriter 评估，您可以使用基准测试 [LongBench-Write](evalscope/third_party/longbench_write/README.md) 来评测长输出的质量以及输出长度。
- 🔥 **[2024.08.30]** 支持自定义数据集评测，包括文本数据集和多模态图文数据集。
- 🔥 **[2024.08.20]** 更新了官方文档，包括快速上手、最佳实践和常见问题等，欢迎[📖阅读](https://evalscope.readthedocs.io/zh-cn/latest/)。
- 🔥 **[2024.08.09]** 简化安装方式，支持pypi安装vlmeval相关依赖；优化多模态模型评估体验，基于OpenAI API方式的评估链路，最高加速10倍。
- 🔥 **[2024.07.31]** 重要修改：`llmuses`包名修改为`evalscope`，请同步修改您的代码。
- 🔥 **[2024.07.26]** 支持**VLMEvalKit**作为第三方评测框架，发起多模态模型评测任务。
- 🔥 **[2024.06.29]** 支持**OpenCompass**作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评估任务配置。
- 🔥 **[2024.06.13]** EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 。
- 🔥 **[2024.06.13]** 接入Agent评测集ToolBench。


## 🛠️ 环境准备
### 方式1. 使用pip安装
我们推荐使用conda来管理环境，并使用pip安装依赖:
1. 创建conda环境 (可选)
```shell
# 建议使用 python 3.10
conda create -n evalscope python=3.10

# 激活conda环境
conda activate evalscope
```
2. pip安装依赖
```shell
pip install evalscope                # 安装 Native backend (默认)
# 额外选项
pip install evalscope[opencompass]   # 安装 OpenCompass backend
pip install evalscope[vlmeval]       # 安装 VLMEvalKit backend
pip install evalscope[all]           # 安装所有 backends (Native, OpenCompass, VLMEvalKit)
```


> [!WARNING]
> 由于项目更名为`evalscope`，对于`v0.4.3`或更早版本，您可以使用以下命令安装：
> ```shell
>  pip install llmuses<=0.4.3
> ```
> 使用`llmuses`导入相关依赖：
> ``` python
> from llmuses import ...
> ```



### 方式2. 使用源码安装
1. 下载源码
```shell
git clone https://github.com/modelscope/evalscope.git
```
2. 安装依赖
```shell
cd evalscope/

pip install -e .                  # 安装 Native backend
# 额外选项
pip install -e '.[opencompass]'   # 安装 OpenCompass backend
pip install -e '.[vlmeval]'       # 安装 VLMEvalKit backend
pip install -e '.[all]'           # 安装所有 backends (Native, OpenCompass, VLMEvalKit)
```


## 🚀 快速开始

### 1. 简单评估
在指定的若干数据集上使用默认配置评估某个模型，流程如下：

#### 使用pip安装

可在任意路径下执行：
```bash
python -m evalscope.run \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --template-type qwen \
 --datasets gsm8k ceval \
 --limit 10
```

#### 使用源码安装

在`evalscope`路径下执行：
```bash
python evalscope/run.py \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --template-type qwen \
 --datasets gsm8k ceval \
 --limit 10
```

> 如遇到 `Do you wish to run the custom code? [y/N]` 请键入 `y`

**运行结果（只使用了10个样例测试）**
```text
Report table:
+-----------------------+--------------------+-----------------+
| Model                 | ceval              | gsm8k           |
+=======================+====================+=================+
| Qwen2.5-0.5B-Instruct | (ceval/acc) 0.5577 | (gsm8k/acc) 0.5 |
+-----------------------+--------------------+-----------------+
```

#### 基本参数说明
- `--model`: 指定了模型在[ModelScope](https://modelscope.cn/)中的`model_id`，可自动下载，例如[Qwen2-0.5B-Instruct模型链接](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct/summary)；也可使用模型的本地路径，例如`/path/to/model`
- `--template-type`: 指定了模型对应的模板类型，参考[模板表格](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html#id4)中的`Default Template`字段填写
- `--datasets`: 数据集名称，支持输入多个数据集，使用空格分开，数据集将自动下载，支持的数据集参考[数据集列表](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset.html)
- `--limit`: 每个数据集最大评估数据量，不填写则默认为全部评估，可用于快速验证评估流程


### 2. 带参数评估
若想进行更加自定义的评估，例如自定义模型参数，或者数据集参数，可以使用以下命令：

**示例1：**
```shell
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --model-args revision=master,precision=torch.float16,device_map=auto \
 --datasets gsm8k ceval \
 --use-cache true \
 --limit 10
```

**示例2：**
```shell
python evalscope/run.py \
 --model qwen/Qwen2-0.5B-Instruct \
 --template-type qwen \
 --generation-config do_sample=false,temperature=0.0 \
 --datasets ceval \
 --dataset-args '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}' \
 --limit 10
```

#### 参数说明
除开上述的[基本参数](#基本参数说明)，其他参数如下：
- `--model-args`: 模型加载参数，以逗号分隔，key=value形式
- `--generation-config`: 生成参数，以逗号分隔，key=value形式
  - `do_sample`: 是否使用采样，默认为`false`
  - `max_new_tokens`: 生成最大长度，默认为1024
  - `temperature`: 采样温度
  - `top_p`: 采样阈值
  - `top_k`: 采样阈值
- `--use-cache`: 是否使用本地缓存，默认为`false`；如果为`true`，则已经评估过的模型和数据集组合将不会再次评估，直接从本地缓存读取
- `--dataset-args`: 评估数据集的设置参数，以json格式传入，key为数据集名称，value为参数，注意需要跟`--datasets`参数中的值一一对应
  - `--few_shot_num`: few-shot的数量
  - `--few_shot_random`: 是否随机采样few-shot数据，如果不设置，则默认为`true`


### 3. 使用run_task函数提交评估任务

使用`run_task`函数提交评估任务所需参数与命令行启动评估任务相同。

需要传入一个字典作为参数，字典中包含以下字段：

#### 1. 配置任务字典参数
```python
import torch
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR

# 示例
your_task_cfg = {
        'model_args': {'revision': None, 'precision': torch.float16, 'device_map': 'auto'},
        'generation_config': {'do_sample': False, 'repetition_penalty': 1.0, 'max_new_tokens': 512},
        'dataset_args': {},
        'dry_run': False,
        'model': 'qwen/Qwen2-0.5B-Instruct',
        'template_type': 'qwen',
        'datasets': ['arc', 'hellaswag'],
        'work_dir': DEFAULT_ROOT_CACHE_DIR,
        'outputs': DEFAULT_ROOT_CACHE_DIR,
        'mem_cache': False,
        'dataset_hub': 'ModelScope',
        'dataset_dir': DEFAULT_ROOT_CACHE_DIR,
        'limit': 10,
        'debug': False
    }
```
其中`DEFAULT_ROOT_CACHE_DIR` 为 `'~/.cache/evalscope'`

#### 2. run_task执行任务
```python
from evalscope.run import run_task

run_task(task_cfg=your_task_cfg)
```

## 使用其他评测后端
EvalScope支持使用第三方评测框架发起评测任务，我们称之为评测后端 (Evaluation Backend)。目前支持的Evaluation Backend有：
- **Native**：EvalScope自身的**默认评测框架**，支持多种评估模式，包括单模型评估、竞技场模式、Baseline模型对比模式等。
- [OpenCompass](https://github.com/open-compass/opencompass)：通过EvalScope作为入口，发起OpenCompass的评测任务，轻量级、易于定制、支持与LLM微调框架[ms-wift](https://github.com/modelscope/swift)的无缝集成：[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)：通过EvalScope作为入口，发起VLMEvalKit的多模态评测任务，支持多种多模态模型和数据集，支持与LLM微调框架[ms-wift](https://github.com/modelscope/swift)的无缝集成：[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html)
- **RAGEval**：通过EvalScope作为入口，发起RAG评测任务，支持使用[MTEB/CMTEB](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/mteb.html)进行embedding模型和reranker的独立评测，以及使用[RAGAS](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html)进行端到端评测：[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/index.html)
- **ThirdParty**: 第三方评估任务，如[ToolBench](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)、[LongBench-Write](https://evalscope.readthedocs.io/zh-cn/latest/third_party/longwriter.html)。

## 推理性能评测工具
一个专注于大型语言模型的压力测试工具，可以自定义以支持各种数据集格式和不同的API协议格式。

参考：性能测试[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test.html)

**支持wandb记录结果**

![wandb sample](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/wandb_sample.png)

**支持Speed Benchmark**

支持速度测试，得到类似[Qwen官方](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html)报告的速度基准：

```text
Speed Benchmark Results:
+---------------+-----------------+----------------+
| Prompt Tokens | Speed(tokens/s) | GPU Memory(GB) |
+---------------+-----------------+----------------+
|       1       |      50.69      |      0.97      |
|     6144      |      51.36      |      1.23      |
|     14336     |      49.93      |      1.59      |
|     30720     |      49.56      |      2.34      |
+---------------+-----------------+----------------+
```



## 自定义数据集评测
EvalScope支持自定义数据集评测，具体请参考：自定义数据集评测[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset.html)

## 离线环境评测
数据集默认托管在[ModelScope](https://modelscope.cn/datasets)上，加载需要联网。如果是无网络环境，参考：离线环境评估[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/offline_evaluation.html)


## 竞技场模式
竞技场模式允许多个候选模型通过两两对比(pairwise battle)的方式进行评估，并可以选择借助AI Enhanced Auto-Reviewer（AAR）自动评估流程或者人工评估的方式，最终得到评估报告。参考：竞技场模式[📖使用指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/arena.html)


## TO-DO List
- [x] RAG evaluation
- [x] VLM evaluation
- [x] Agents evaluation
- [x] vLLM
- [ ] Distributed evaluating
- [x] Multi-modal evaluation
- [ ] Benchmarks
  - [ ] GAIA
  - [ ] GPQA
  - [x] MBPP
- [ ] Auto-reviewer
  - [ ] Qwen-max


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/evalscope&type=Date)](https://star-history.com/#modelscope/evalscope&Date)
