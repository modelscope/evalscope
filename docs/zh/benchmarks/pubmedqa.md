# PubMedQA

## 概述

PubMedQA 是一个生物医学问答数据集，旨在评估模型在生物医学研究文本上的推理能力。该数据集包含从 PubMed 摘要中提取的问题，并提供 yes/no/maybe 三类答案。

## 任务描述

- **任务类型**：生物医学问答（Yes/No/Maybe）
- **输入**：包含研究问题的 PubMed 摘要
- **输出**：答案（YES、NO 或 MAYBE）
- **领域**：生物医学研究、科学文献

## 主要特点

- 问题来源于真实的 PubMed 研究摘要
- 三分类任务（YES/NO/MAYBE）
- 考察科学推理与理解能力
- 由专家标注并附带推理解释
- 适用于生物医学 AI 的评估

## 评估说明

- 默认配置使用 **0-shot** 评估
- 评估指标：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、YES 比例（Yes Ratio）、MAYBE 比例（Maybe Ratio）
- 聚合方法：F1 分数（宏平均）
- 答案应仅为 YES、NO 或 MAYBE，无需附加解释

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `pubmedqa` |
| **数据集ID** | [extraordinarylab/pubmed-qa](https://modelscope.cn/datasets/extraordinarylab/pubmed-qa/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `Yes/No` |
| **指标** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio`, `maybe_ratio` |
| **默认示例数量** | 0-shot |
| **评估分割** | `test` |
| **聚合方式** | `f1` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（平均） | 1514.46 字符 |
| 提示词长度（最小/最大） | 451 / 2883 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "f17dedfd",
      "content": [
        {
          "text": "Abstract: Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longit ... [TRUNCATED] ... ntrols, and that displayed mitochondrial dynamics similar to that of non-PCD cells.\n\nQuestion: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\nPlease answer YES or NO or MAYBE without an explanation."
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "yes",
    "reasoning": "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. To the best of our knowledge, this is the first report of ... [TRUNCATED] ... PCD. Also, for the first time, we have shown the feasibility for the use of CsA in a whole plant system. Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant."
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
Please answer YES or NO or MAYBE without an explanation.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets pubmedqa \
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
    datasets=['pubmedqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```