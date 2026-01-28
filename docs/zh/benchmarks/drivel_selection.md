# DrivelologyNarrativeSelection

## 概述

Drivelology 叙事选择（Drivelology Narrative Selection）用于评估模型理解“drivelology”文本底层叙事的能力——这类语言表达在句法上连贯，但在语用层面具有悖论性、情感负载强烈或修辞上具有颠覆性。

## 任务描述

- **任务类型**：多项选择式叙事理解
- **输入**：包含多个叙事解释选项的 drivelology 文本
- **输出**：最能代表文本底层叙事的最佳选项
- **领域**：语言分析、叙事理解

## 核心特点

- 考察深层叙事理解能力
- 要求解读多层次含义
- 多项选择格式，干扰项具有挑战性
- 包含简单和困难两个难度级别
- 考察文化与语境理解能力

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用简单准确率（accuracy）作为指标
- 子集：`multiple-choice-english-easy`、`multiple-choice-english-hard`

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `drivel_selection` |
| **数据集ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **论文** | N/A |
| **标签** | `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,200 |
| 提示词长度（平均） | 1413.26 字符 |
| 提示词长度（最小/最大） | 754 / 2865 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `multiple-choice-english-easy` | 600 | 1563.53 | 908 | 2865 |
| `multiple-choice-english-hard` | 600 | 1262.98 | 754 | 2348 |

## 样例示例

**子集**: `multiple-choice-english-easy`

```json
{
  "input": [
    {
      "id": "44908073",
      "content": "Tell me the best option in the following options which represents the underlying narrative of the text?\nThe entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C, ... [TRUNCATED] ...  be achieved by simply resting today and tomorrow. It humorously implies that diligence is overrated and taking breaks is the key to progress. This narrative undermines the importance of effort, presenting relaxation as the ultimate solution."
    }
  ],
  "choices": [
    "The passage reflects on the inevitability of fate, stating that what happens the day after tomorrow is beyond our control. Therefore, it encourages living in the moment and enjoying today and tomorrow. It conveys a message of mindfulness and acceptance.",
    "This creates a paradoxical tone, as it acknowledges the value of diligence but simultaneously advocates for procrastination. The underlying message could reflect a lighthearted take on balancing work and rest or even poking fun at the tendency to delay responsibilities.",
    "The text discusses the cyclical nature of time, arguing that the day after tomorrow holds the key to breaking free from monotony. It proposes resting today and tomorrow to prepare for this transformative moment. This symbolizes renewal and the anticipation of change.",
    "The text emphasizes the importance of teamwork, suggesting that collective effort tomorrow will yield the best results. It then humorously advises everyone to take a break today to gather energy. This highlights the value of preparation over immediate action.",
    "The text suggests that hard work is unnecessary, as success can be achieved by simply resting today and tomorrow. It humorously implies that diligence is overrated and taking breaks is the key to progress. This narrative undermines the importance of effort, presenting relaxation as the ultimate solution."
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Tell me the best option in the following options which represents the underlying narrative of the text?
The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drivel_selection \
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
    datasets=['drivel_selection'],
    dataset_args={
        'drivel_selection': {
            # subset_list: ['multiple-choice-english-easy', 'multiple-choice-english-hard']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```