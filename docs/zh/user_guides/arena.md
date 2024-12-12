# 竞技场模式
竞技场模式允许多个候选模型通过两两对比(pairwise battle)的方式进行评测，并可以选择借助AI Enhanced Auto-Reviewer（AAR）自动评测流程或者人工评测的方式，最终得到评测报告，本框架支持如下三种模型评测流程：

## 全部模型两两对比（Pairwise mode）
### 1. 环境准备
```text
a. 数据准备，questions data格式参考：evalscope/registry/data/question.jsonl
b. 如果需要使用自动评测流程（AAR），则需要配置相关环境变量，我们以GPT-4 based auto-reviewer流程为例，需要配置以下环境变量：
> export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

### 2. 配置文件
```text
arena评测流程的配置文件参考： evalscope/registry/config/cfg_arena.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评测结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    elo_rating: ELO rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```

### 3. 执行脚本
```shell
#Usage:
cd evalscope

# dry-run模式 (模型answer正常生成，但专家模型，如GPT-4，不会被调用，评测结果会随机生成)
python evalscope/run_arena.py -c registry/config/cfg_arena.yaml --dry-run

# 执行评测流程
python evalscope/run_arena.py --c registry/config/cfg_arena.yaml
```

### 4. 结果可视化

```shell
# Usage:
streamlit run viz.py --review-file evalscope/registry/data/qa_browser/battle.jsonl --category-file evalscope/registry/data/qa_browser/category_mapping.yaml
```



## 单模型打分模式（Single mode）

这个模式下，我们只对单个模型输出做打分，不做两两对比。
### 1. 配置文件
```text
评测流程的配置文件参考： evalscope/registry/config/cfg_single.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评测结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```
### 2. 执行脚本
```shell
#Example:
python evalscope/run_arena.py --c registry/config/cfg_single.yaml
```

## Baseline模型对比模式（Pairwise-baseline mode）

这个模式下，我们选定 baseline 模型，其他模型与 baseline 模型做对比评分。这个模式可以方便的把新模型加入到 Leaderboard 中（只需要对新模型跟 baseline 模型跑一遍打分即可）
### 1. 配置文件
```text
评测流程的配置文件参考： evalscope/registry/config/cfg_pairwise_baseline.yaml
字段说明：
    questions_file: question data的路径
    answers_gen: 候选模型预测结果生成，支持多个模型，可通过enable参数控制是否开启该模型
    reviews_gen: 评测结果生成，目前默认使用GPT-4作为Auto-reviewer，可通过enable参数控制是否开启该步骤
    rating_gen: rating 算法，可通过enable参数控制是否开启该步骤，注意该步骤依赖review_file必须存在
```
### 2. 执行脚本
```shell
# Example:
python evalscope/run_arena.py --c registry/config/cfg_pairwise_baseline.yaml
```