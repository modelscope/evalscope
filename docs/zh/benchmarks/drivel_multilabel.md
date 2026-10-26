# DrivelologyMultilabelClassification

## 概述

Drivelology 多标签分类评估模型将“drivelology”文本分类为修辞技巧类别的能力：倒装、双关、标题党、悖论和误导。每段文本可能属于多个类别。

## 任务描述

- **任务类型**：多标签文本分类
- **输入**：Drivelology 文本样本
- **输出**：一个或多个技巧类别
- **领域**：语言分析，修辞技巧检测

## 主要特点

- 五个修辞技巧类别
- 多标签分类（每个样本可以有多个类别）
- 测试对语言创造机制的理解
- 需要识别幽默和讽刺技巧
- 提供详细的类别定义

## 评估说明

- 默认配置使用 **0-shot** 评估
- 指标：F1（加权、微观、宏观）、完全匹配
- 最后一行 `ANSWER:` 上的标签按列出的方式读取：`AD`，`A, D` 或 `A and D`
- 聚合方法：加权 F1
- 类别：倒装、双关、标题党、悖论、误导

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `drivel_multilabel` |
| **数据集ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **论文** | N/A |
| **标签** | `MCQ` |
| **指标** | `f1_weighted`, `f1_micro`, `f1_macro`, `exact_match` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |
| **聚合** | `f1_weighted` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 600 |
| 提示词长度（平均） | 1637.18 字符 |
| 提示词长度（最小/最大） | 1580 / 2041 字符 |

## 样例示例

**子集**: `multi-label-classification`

```json
{
  "input": [
    {
      "id": "799dcfaa",
      "content": [
        {
          "text": "#Instruction#:\nClassify the given text into one or more of the following categories: inversion, wordplay, switchbait, paradox, and misdirection.\n\n#Definitions#:\n- inversion: This technique takes a well-known phrase, cliché, or social script a ... [TRUNCATED 1108 chars] ... e should be of the following format: 'ANSWER: [LETTERS]' (without quotes) where [LETTER]S is one or more of A,B,C,D,E.\n\nText to classify: 後天的努力比什麼都重要，所以今天和明天休息。\n\nA) A. inversion\nB) B. wordplay\nC) C. switchbait\nD) D. paradox\nE) E. misdirection"
        }
      ]
    }
  ],
  "choices": [
    "A. inversion",
    "B. wordplay",
    "C. switchbait",
    "D. paradox",
    "E. misdirection"
  ],
  "target": "AB",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "text": "後天的努力比什麼都重要，所以今天和明天休息。",
    "label": [
      "inversion",
      "wordplay"
    ],
    "target_letters": "AB"
  }
}
```

*注：部分内容为了显示被截断。*

## 提示模板

**提示模板:**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drivel_multilabel \
    --limit 10  # Remove this line for formal evaluation
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['drivel_multilabel'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
