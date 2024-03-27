## 简介
大型语言模型评估（LLMs evaluation）已成为评价和改进大模型的重要流程和手段，为了更好地支持大模型的评测，我们提出了llmuses框架，该框架主要包括以下几个部分：
- 预置了多个常用的测试基准数据集，包括：MMLU、C-Eval、GSM8K、ARC、HellaSwag、TruthfulQA、MATH、HumanEval等
- 常用评估指标（metrics）的实现
- 统一model接入，兼容多个系列模型的generate、chat接口
- 自动评估（evaluator）：
    - 客观题自动评估
    - 使用专家模型实现复杂任务的自动评估
- 评估报告生成
- 竞技场模式(Arena）
- 可视化工具
- [模型性能评估](llmuses/perf/README.md)

特点
- 轻量化，尽量减少不必要的抽象和配置
- 易于定制
  - 仅需实现一个类即可接入新的数据集
  - 模型可部署在本地，或[ModelScope](https://modelscope.cn)上
  - 评估报告可视化展现
- 丰富的评估指标
- model-based自动评估流程，支持多种评估模式
  - Single mode: 专家模型对单个模型打分
  - Pairwise-baseline mode: 与 baseline 模型对比
  - Pairwise (all) mode: 全部模型两两对比


## 环境准备
```shell
# 1. 代码下载
git clone git@github.com:modelscope/llmuses.git

# 2. 安装依赖
cd llmuses/
pip install -r requirements/requirements.txt
pip install -e .

# Note: 您也可以使用自定义的源安装依赖
pip install -r requirements/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```


## 快速开始

### 简单评估
```shell
# 在特定数据集上评估某个模型
python llmuses/run.py --model ZhipuAI/chatglm3-6b --datasets mmlu ceval --limit 10
```

### 带参数评估
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --model-args revision=v1.0.2,precision=torch.float16,device_map=auto --datasets mmlu ceval --mem-cache --limit 10

python llmuses/run.py --model qwen/Qwen-1_8B --generation-config do_sample=false,temperature=0.0 --datasets ceval --dataset-args '{"ceval": {"few_shot_num": 0, "few_shot_random": false}}' --limit 10

# 参数说明
# --model-args: 模型参数，以逗号分隔，key=value形式
# --datasets: 数据集名称，参考下文`数据集列表`章节
# --mem-cache: 是否使用内存缓存，若开启，则已经跑过的数据会自动缓存，并持久化到本地磁盘
# --limit: 每个subset最大评估数据量
# --dataset-args: 数据集的evaluation settings，以json格式传入，key为数据集名称，value为参数，注意需要跟--datasets参数中的值一一对应
#   -- few_shot_num: few-shot的数量
#   -- few_shot_random: 是否随机采样few-shot数据，如果不设置，则默认为true
```

### 使用本地数据集
数据集默认托管在[ModelScope](https://modelscope.cn/datasets)上，加载需要联网。如果是无网络环境，可以使用本地数据集，流程如下：
#### 1. 下载数据集到本地
```shell
# 假如当前本地工作路径为 /path/to/workdir
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/benchmark/data.zip
unzip data.zip
# 则解压后的数据集路径为：/path/to/workdir/data 目录下，该目录在后续步骤将会作为--dataset-dir参数的值传入
```
#### 2. 使用本地数据集创建评估任务
```shell
python llmuses/run.py --model ZhipuAI/chatglm3-6b --datasets arc --dataset-hub Local --dataset-dir /path/to/workdir/data --limit 10

# 参数说明
# --dataset-hub: 数据集来源，枚举值： `ModelScope`, `Local`, `HuggingFace` (TO-DO)  默认为`ModelScope`
# --dataset-dir: 当--dataset-hub为`Local`时，该参数指本地数据集路径; 如果--dataset-hub 设置为`ModelScope` or `HuggingFace`，则该参数的含义是数据集缓存路径。

```
#### 3. (可选)在离线环境加载模型和评测
模型文件托管在ModelScope Hub端，需要联网加载，当需要在离线环境创建评估任务时，可参考以下步骤：
```shell
# 1. 准备模型本地文件夹，文件夹结构参考[chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/files)
# 例如，将模型文件夹整体下载到本地路径 /path/to/ZhipuAI/chatglm3-6b

# 2. 执行离线评估任务
python llmuses/run.py --model /path/to/ZhipuAI/chatglm3-6b --datasets arc --dataset-hub Local --dataset-dir /path/to/workdir/data --limit 10

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

# dry-run模式 (模型answer正常生成，但专家模型不会被触发，评估结果会随机生成)
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
python llmuses/run_arena.py --c llmuses/registry/config/cfg_pairwise_baseline.yaml
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
