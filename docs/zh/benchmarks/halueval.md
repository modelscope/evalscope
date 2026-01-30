# HaluEval

## 概述

HaluEval 是一个大规模的生成与人工标注的幻觉样本集合，用于评估大语言模型（LLM）识别幻觉的能力。它提供了一个全面的基准测试，用于衡量模型的可靠性和事实准确性。

## 任务描述

- **任务类型**：幻觉检测
- **输入**：上下文/知识 + 待判断的回复
- **输出**：YES（存在幻觉）或 NO（符合事实）
- **领域**：对话、问答、摘要

## 主要特点

- 包含三个评估类别：
  - `dialogue_samples`：对话回复中的幻觉
  - `qa_samples`：问答中的幻觉
  - `summarization_samples`：文档摘要中的幻觉
- 同时包含生成样本和人工标注样本
- 测试模型检测事实不一致性的能力
- 要求模型对知识与回复之间的一致性进行推理

## 评估说明

- 默认评估采用 **零样本（zero-shot）** 设置（无少样本示例）
- 计算多项指标：
  - **Accuracy（准确率）**：整体判断正确的比例
  - **Precision（精确率）**：预测为正类中真实为正的比例
  - **Recall（召回率）**：真实为正类中被正确预测的比例
  - **F1 Score（F1分数）**：精确率与召回率的调和平均
  - **Yes Ratio（YES比例）**：预测为 YES 的比例
- 采用二元 YES/NO 判断格式

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `halueval` |
| **数据集ID** | [evalscope/HaluEval](https://modelscope.cn/datasets/evalscope/HaluEval/summary) |
| **论文** | N/A |
| **标签** | `Hallucination`, `Knowledge`, `Yes/No` |
| **指标** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **默认样本数** | 0-shot |
| **评估划分** | `data` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 30,000 |
| 提示词长度（平均） | 4832.18 字符 |
| 提示词长度（最小/最大） | 2463 / 16078 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `dialogue_samples` | 10,000 | 3563.69 | 3169 | 4200 |
| `qa_samples` | 10,000 | 2811.83 | 2463 | 4004 |
| `summarization_samples` | 10,000 | 8121.02 | 4932 | 16078 |

## 样例示例

**子集**：`dialogue_samples`

```json
{
  "input": [
    {
      "id": "a99406f3",
      "content": [
        {
          "text": "I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallu ... [TRUNCATED] ...  do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film. \n#Response#: I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.\n#Your Judgement#:"
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "yes"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets halueval \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['halueval'],
    dataset_args={
        'halueval': {
            # subset_list: ['dialogue_samples', 'qa_samples', 'summarization_samples']  # 可选，用于指定评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```