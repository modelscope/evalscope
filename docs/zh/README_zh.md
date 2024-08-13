[English](README.md) | 简体中文

![](_static/images/evalscope.jpeg)

<p align="center">
<a href="https://pypi.org/project/evalscope"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/evalscope">
</a>
<a href="https://github.com/modelscope/evalscope/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
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
大型语言模型评估（LLMs evaluation）已成为评价和改进大模型的重要流程和手段，为了更好地支持大模型的评测，我们提出了EvalScope框架，该框架主要包括以下几个部分：

![](_static/images/evalscope_framework.png)
*EvalScope 架构图.*

- 预置了多个常用的测试基准数据集，包括：MMLU、CMMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH、HumanEval等
- 常用评估指标（metrics）的实现
- 统一model接入，兼容多个系列模型的generate、chat接口
- 自动评估（evaluator）：
    - 客观题自动评估
    - 使用专家模型实现复杂任务的自动评估
- 评估报告生成
- 竞技场模式(Arena)
- 可视化工具
- [模型性能评估](evalscope/perf/README.md)
- 支持OpenCompass作为评测后段，对其进行了高级封装和任务简化，您可以更轻松地提交任务到OpenCompass进行评估。
- 支持VLMEvalKit作为评测后端，通过EvalScope作为入口，发起VLMEvalKit的多模态评测任务，支持多种多模态模型和数据集。
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
- **[2024.07.31]** 重要修改：`llmuses`包名修改为`evalscope`，请同步修改您的代码
- **[2024.07.26]** 支持**VLMEvalKit**作为第三方评测框架，发起多模态模型评测任务，[使用指南](#vlmevalkit-评测后端) 🔥🔥🔥
- **[2024.06.29]** 支持**OpenCompass**作为第三方评测框架，我们对其进行了高级封装，支持pip方式安装，简化了评估任务配置，[使用指南](#opencompass-评测后端) 🔥🔥🔥
- **[2024.06.13]** EvalScope与微调框架SWIFT进行无缝对接，提供LLM从训练到评测的全链路支持 🚀🚀🚀
- **[2024.06.13]** 接入Agent评测集ToolBench 🚀🚀🚀

### 使用评测后端 (Evaluation Backend)
EvalScope支持使用第三方评测框架发起评测任务，我们称之为评测后端 (Evaluation Backend)。目前支持的Evaluation Backend有：
- **Native**：EvalScope自身的**默认评测框架**，支持多种评估模式，包括单模型评估、竞技场模式、Baseline模型对比模式等。
- [OpenCompass](https://github.com/open-compass/opencompass)：通过EvalScope作为入口，发起OpenCompass的评测任务，轻量级、易于定制、支持与LLM微调框架[ModelScope Swift](https://github.com/modelscope/swift)的无缝集成。
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)：通过EvalScope作为入口，发起VLMEvalKit的多模态评测任务，支持多种多模态模型和数据集，支持与LLM微调框架[ModelScope Swift](https://github.com/modelscope/swift)的无缝集成。
- **ThirdParty**: 第三方评估任务，如[ToolBench](evalscope/thirdparty/toolbench/README.md)。


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
python evalscope/run.py --model ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --dataset-hub Local --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' --limit 10

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
python evalscope/run.py --model /path/to/ZhipuAI/chatglm3-6b --template-type chatglm3 --datasets arc --dataset-hub Local --dataset-args '{"arc": {"local_path": "/path/to/workdir/data/arc"}}' --limit 10
```


### 使用run_task函数提交评估任务

#### 1. 配置任务
```python
import torch
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR

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
from evalscope.run import run_task

run_task(task_cfg=your_task_cfg)
```


### 竞技场模式（Arena）
竞技场模式允许多个候选模型通过两两对比(pairwise battle)的方式进行评估，并可以选择借助AI Enhanced Auto-Reviewer（AAR）自动评估流程或者人工评估的方式，最终得到评估报告，流程示例如下：
#### 1. 环境准备
```text
a. 数据准备，questions data格式参考：evalscope/registry/data/question.jsonl
b. 如果需要使用自动评估流程（AAR），则需要配置相关环境变量，我们以GPT-4 based auto-reviewer流程为例，需要配置以下环境变量：
> export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

#### 2. 配置文件
```text
arena评估流程的配置文件参考： evalscope/registry/config/cfg_arena.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    elo_rating: ELO rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```

#### 3. 执行脚本
```shell
#Usage:
cd evalscope

# dry-run模式 (模型answer正常生成，但专家模型，如GPT-4，不会被调用，评估结果会随机生成)
python evalscope/run_arena.py -c registry/config/cfg_arena.yaml --dry-run

# 执行评估流程
python evalscope/run_arena.py --c registry/config/cfg_arena.yaml
```

#### 4. 结果可视化

```shell
# Usage:
streamlit run viz.py --review-file evalscope/registry/data/qa_browser/battle.jsonl --category-file evalscope/registry/data/qa_browser/category_mapping.yaml
```


### 单模型打分模式（Single mode）

这个模式下，我们只对单个模型输出做打分，不做两两对比。
#### 1. 配置文件
```text
评估流程的配置文件参考： evalscope/registry/config/cfg_single.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```
#### 2. 执行脚本
```shell
#Example:
python evalscope/run_arena.py --c registry/config/cfg_single.yaml
```

### Baseline模型对比模式（Pairwise-baseline mode）

这个模式下，我们选定 baseline 模型，其他模型与 baseline 模型做对比评分。这个模式可以方便的把新模型加入到 Leaderboard 中（只需要对新模型跟 baseline 模型跑一遍打分即可）
#### 1. 配置文件
```text
评估流程的配置文件参考： evalscope/registry/config/cfg_pairwise_baseline.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```
#### 2. 执行脚本
```shell
# Example:
python evalscope/run_arena.py --c registry/config/cfg_pairwise_baseline.yaml
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
参考： [性能测试](evalscope/perf/README.md)

## TO-DO List
- [x] Agents evaluation
- [ ] vLLM
- [ ] Distributed evaluating
- [x] Multi-modal evaluation
- [ ] Benchmarks
  - [ ] GAIA
  - [ ] GPQA
  - [x] MBPP
- [ ] Auto-reviewer
  - [ ] Qwen-max


