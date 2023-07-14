# 基本功能介绍
大型语言模型评估（LLMs evaluation）已成为评价和改进大模型的重要流程和手段，为了更好地支持大模型的评测，我们提出了llmuses框架，该框架主要包括以下几个部分：
- 评估数据集的注册、管理和预处理
- 评估方法（打分模型，scoring model）的实现
- 评估指标（metrics）的实现
- 抽象了评估任务（eval task），通过配置文件实现对评估方法、数据、metrics的管理和串联
- 评估报告生成
- 自动评估流程和人工评估流程的支持
- 竞技场模式(Arena），支持多个模型的两两对比评估
- AI Enhanced Auto-Reviewer（AAR），支持基于专家模型的自动评估流程

我们提供以下评估指标：
1. rouge
2. reward model
3. math accuracy，评估方法为获取基础模型生成的最后的数字，和target对比，也就是只考虑结果分，不考虑过程分。
4. code pass@k，评估方法只支持Python，每个题目生成有随机性的k个答案，只要通过一个即算通过。
5. 另外，本框架亦支持基于iTAG人工评估流程。

llmuses框架还支持其他LLM的接口，包括：
- 千问dashscope接口（暂未开放）
- ChatGLM
- OpenAI gpt-3.5-turbo、元语等需要token的LLM服务（陆续支持）


# 快速开始
## Auto-Evaluate

Auto-Evaluate指无需人工干预的自动评估任务，其输入的数据中已包含模型预测结果和ground truth，以下是几个自动评估的任务示例：
#### 1. 模型coding能力评估
```python
python3 scripts/run_eval.py --input evals/registry/data/code/code_test_v2_model_result.jsonl --task_cfg evals/registry/tasks/task_qwen_code.yaml --eval-type=code
```

#### 2. 模型数学能力评估
```python
python3 scripts/run_eval.py --input evals/registry/data/math/math_test_v2_model_result.jsonl --task_cfg evals/registry/tasks/task_qwen_math.yaml --eval-type=math
```

#### 3. 模型通用生成能力评估（翻译、诗歌生成、成语接龙等）
```python
python3 scripts/run_eval.py --input evals/registry/data/common_generation/rouge_test_v7_model_result.jsonl --task_cfg evals/registry/tasks/task_qwen_generation.yaml --eval-type=rouge
```


## 竞技场模式（Arena）
竞技场模式允许多个候选模型通过两两对比(pairwise battle)的方式进行评估，并可以选择借助AI Enhanced Auto-Reviewer（AAR）自动评估流程或者人工评估的方式，最终得到评估报告，流程示例如下：
#### 1. 环境准备
```text
a. 数据准备，questions data格式参考：evals/registry/data/arena/question.jsonl
b. 如果需要使用自动评估流程（AAR），则需要配置相关环境变量，我们以GPT-4 based auto-reviewer流程为例，需要配置以下环境变量：
> export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

#### 2. 配置文件
```text
arena评估流程的配置文件参考： evals/registry/tasks/cfg_arena.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    elo_rating: ELO rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```

#### 3. 执行脚本
```python
Usage:
# python3 scripts/run_arena.py --c /path/to/xxx_cfg_arena.yaml

Example:
> python3 scripts/run_arena.py --c evals/registry/tasks/cfg_arena.yaml
```

#### 4. 结果可视化

```python
Usage:
# streamlit run apps/app.py -- --review_file /path/to/xxx_review_file.jsonl --category_file /path/to/xxx_category_mapping.yaml

Example:
> streamlit run scripts/run_qa_browser.py -- --review_file evals/registry/data/qa_browser/battle.jsonl --category_file evals/registry/data/qa_browser/category_mapping.yaml
```

### 其他的评分模式
除了竞技场模式两两对比（pairwise battle）的评分模式，我们还支持以下两种评分模式：
- `single`: 只对单个模型输出做打分，不做两两对比
- `pairwise-baseline`: 选定 baseline 模型，其他模型与 baseline 模型做对比评分

#### 1. Single mode: 单个模型打分

这个模式下，我们只对单个模型输出做打分，不做两两对比。这个模式可以更方便的把新模型加入到 Leaderboard 中（只需要对新模型跑一遍打分即可）

```text
评估流程的配置文件参考： evals/registry/tasks/cfg_single.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```

```python
Example:
> python3 scripts/run_arena.py --c evals/registry/tasks/cfg_single.yaml
```

#### 2. Pairwise-baseline mode: 与 baseline 模型对比

这个模式下，我们选定 baseline 模型，其他模型与 baseline 模型做对比评分。这个模式可以方便的把新模型加入到 Leaderboard 中（只需要对新模型跟 baseline 模型跑一遍打分即可）

```text
评估流程的配置文件参考： evals/registry/tasks/cfg_pairwise_baseline.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评估结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```

```python
Example:
> python3 scripts/run_arena.py --c evals/registry/tasks/cfg_pairwise_baseline.yaml
```

# 数据格式

为了保证能够方便的贡献评测数据集，我们设计了最基础的数据格式和为了相应评估方法的可选数据格式。 数据是以jsonl的形式存储，具体介绍如下：

`注`：输入部分是基础模型的输入，可以通过本repo得到（如果已经支持该基础模型）,也可以自己通过其他任意方式得到结果。

## 输入格式

| key | 类型 | 是否必须 | 含义 |
| --- | --- | --- | ---|
| `id`| 字符串 | 是 | 当前prompt id|
| `prompt` | 字符串 | 是 | 当前prompt，可以认为是用户最近如输入的信息，如`请问中国的首都是哪里？` |
| `history`| list of dict | 否 | 如：[{"human": "成语接龙，无可厚非", "assistant": "非亲非故"}]|
| `target` | list of str | 否 | 如果不需要ground truth就能评估，list的原因是可以有多个ground truth |
| `func_args`| list | 否 | 针对代码生成任务的评估，这个是函数的输入参数 |
| `func_outputs` | list | 否 | 针对代码生成任务的输入，这是正确的输出 |
| `task_tags` | list of str | 否 | 当前prompt所属的task, 不唯一 |
| `ability_tags` | list of str | 否 | 能力维度的标签，不唯一 |
| `industry_tags` | list of str | 否 | 行业维度的标签，不唯一 |
| `level_tags` | 字符串 | 否 | 难度维度的标签，唯一 |

## 输出格式
keep all keys of the input json, and add a new key named `gen`

| key | 类型 | 是否必须 | 含义 |
| --- | --- | --- | ---|
| `gen`| list of str | 是 | 生成结果，可以生成多个，所以这里是list |
