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
```


## 快速开始

### 简单评估
```shell
# 在特定数据集上评估某个模型
python run.py --model ZhipuAI/chatglm3-6b --datasets mmlu ceval --limit 10
```

### 带参数评估
```shell
python run.py --model ZhipuAI/chatglm3-6b --model-args revision=v1.0.2,precision=torch.float16,device_map=auto --datasets mmlu ceval --mem-cache --limit 10

# 参数说明
# --model-args: 模型参数，以逗号分隔，key=value形式
# --datasets: 数据集名称，参考下文`数据集列表`章节
# --mem-cache: 是否使用内存缓存，若开启，则已经跑过的数据会自动缓存，并持久化到本地磁盘
# --limit: 每个subset最大评估数据量
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
python run_arena.py -c llmuses/registry/config/cfg_arena.yaml --dry-run

# 执行评估流程
python run_arena.py --c llmuses/registry/config/cfg_arena.yaml
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
python run_arena.py --c llmuses/registry/config/cfg_single.yaml
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
python run_arena.py --c llmuses/registry/config/cfg_pairwise_baseline.yaml
```


## 数据集列表

| dataset name       | link                                                                                   | status | note |
|--------------------|----------------------------------------------------------------------------------------|--------|------|
| `mmlu`             | [mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)                         | active |    |
| `ceval`            | [ceval](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)                  | active |    |
| `gsm8k`            | [gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)                       | active |    |
| `arc`              | [arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)                       | active |    |
| `hellaswag`        | [hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)               | active |    |
| `truthful_qa`      | [truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)           | active |    |
| `competition_math` | [competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary) | active |    |
| `humaneval`        | [humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)               | active |    |


## Leaderboard 榜单
ModelScope LLM Leaderboard大模型评测榜单旨在提供一个客观、全面的评估标准和平台，帮助研究人员和开发者了解和比较ModelScope上的模型在各种任务上的性能表现。

[Leaderboard](https://modelscope.cn/leaderboard/58/ranking?type=free)

## 实验和报告

### [MMLU](https://modelscope.cn/datasets/modelscope/mmlu/summary)

##### Settings: (Split: test, Total num: 13985, 0-shot)

| Model                                                                                            | Revision | Precision | Humanities  | STEM       | SocialScience | Other   | WeightedAvg | Target      | \Delta |
|--------------------------------------------------------------------------------------------------|----------|-----------|-------------|------------|---------------|---------|-------------|-------------|--------|
| [Baichuan2-7B-Base](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary)         | v1.0.2   | fp16      | 0.4111      | 0.3807     | 0.5233        | 0.504   | 0.4506      | -           |        |
| [Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-chat/summary)         | v1.0.4   | fp16      | 0.4439      | 0.374      | 0.5524        | 0.5458  | 0.4762      | -           |        |
| [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary)                          | v1.0.12  | fp16      | 0.3834      | 0.3413     | 0.4708        | 0.4445  | 0.4077      | 0.4546(CoT) | -4.69% |
| [chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base/summary)                | v1.0.1   | fp16      | 0.5435      | 0.5087     | 0.7227        | 0.6471  | 0.5992      | 0.614       | 1.48%  |
| [internlm-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b/summary) | v1.0.1   | fp16      | 0.4005      | 0.3547     | 0.4953        | 0.4796  | 0.4297      | -           |        |
| [Llama-2-13b-ms](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary)                 | v1.0.2   | fp16      | 0.4371      | 0.3887     | 0.5579        | 0.5437  | 0.4778      | -           |        |
| [Llama-2-7b-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary)                   | v1.0.2   | fp16      | 0.3146      | 0.3037     | 0.4134        | 0.3885  | 0.3509      | -           |        |
| [Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary)                         | v1.0.6   | bf16      | 0.5326      | 0.5397     | 0.7184        | 0.6859  | 0.6102      | -           |        |
| [Qwen-7B](https://modelscope.cn/models/qwen/Qwen-7B/summary)                                     | v1.1.6   | bf16      | 0.387       | 0.4        | 0.5403        | 0.5139  | 0.4527      | -           |        |
| [Qwen-7B-Chat-Int8](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary)                 | v1.1.6   | int8      | 0.4322      | 0.4277     | 0.6088        | 0.5778  | 0.5035      | -           |        |


##### Settings: (Split: test, Total num: 13985, 5-shot)

| Model               | Revision | Precision | Humanities | STEM   | SocialScience | Other  | WeightedAvg | Avg    | Target             | \Delta   |
|---------------------|----------|-----------|------------|--------|---------------|--------|-------------|--------|--------------------|----------|
| Baichuan2-7B-Base   | v1.0.2   | fp16      | 0.4295     | 0.398  | 0.5736        | 0.5325 | 0.4781      | 0.4918 | 0.5416 (official)  | -4.98%   |
| Baichuan2-7B-Chat   | v1.0.4   | fp16      | 0.4344     | 0.3937 | 0.5814        | 0.5462 | 0.4837      | 0.5029 | 0.5293 (official)  | -2.64%   |
| chatglm2-6b         | v1.0.12  | fp16      | 0.3941     | 0.376  | 0.4897        | 0.4706 | 0.4288      | 0.4442 | -                  | -        |
| chatglm3-6b-base    | v1.0.1   | fp16      | 0.5356     | 0.4847 | 0.7175        | 0.6273 | 0.5857      | 0.5995 | -                  | -        |
| internlm-chat-7b    | v1.0.1   | fp16      | 0.4171     | 0.3903 | 0.5772        | 0.5493 | 0.4769      | 0.4876 | -                  | -        |
| Llama-2-13b-ms      | v1.0.2   | fp16      | 0.484      | 0.4133 | 0.6157        | 0.5809 | 0.5201      | 0.5327 | 0.548 (official)   | -1.53%   |
| Llama-2-7b-ms       | v1.0.2   | fp16      | 0.3747     | 0.3363 | 0.4372        | 0.4514 | 0.3979      | 0.4089 | 0.453 (official)   | -4.41%   |
| Qwen-14B-Chat       | v1.0.6   | bf16      | 0.574      | 0.553  | 0.7403        | 0.684  | 0.6313      | 0.6414 | 0.646 (official)   | -0.46%   |
| Qwen-7B             | v1.1.6   | bf16      | 0.4587     | 0.426  | 0.6078        | 0.5629 | 0.5084      | 0.5151 | 0.567 (official)   | -5.2%    |
| Qwen-7B-Chat-Int8   | v1.1.6   | int8      | 0.4697     | 0.4383 | 0.6284        | 0.5967 | 0.5271      | 0.5347 | 0.554 (official)   | -1.93%   |



## TO-DO List
- [ ] Agents evaluation
- [ ] vLLM
- [ ] Distributed evaluating
- [ ] Multi-modal evaluation
- [ ] Benchmarks
  - [ ] GAIA
  - [ ] GPQA
  - [ ] BBH
  - [ ] MBPP
- [ ] Auto-reviewer
  - [ ] Qwen-max
