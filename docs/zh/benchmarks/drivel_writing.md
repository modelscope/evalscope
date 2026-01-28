# DrivelologyNarrativeWriting

## 概述

Drivelology 叙事写作评估模型生成详细描述的能力，以阐释“drivelology”文本中隐含的叙事——这类语言表达在句法上连贯，但在语用上却具有悖论性、情感负载性或修辞颠覆性。

## 任务描述

- **任务类型**：叙事生成与评估  
- **输入**：Drivelology 文本样本  
- **输出**：生成的叙事描述，解释文本的隐含意义  
- **领域**：语言学分析、叙事生成  

## 核心特点

- 测试模型生成叙事解释的能力  
- 要求理解多层次的语言含义  
- 使用 LLM-as-judge 方法与参考叙事进行对比评估  
- 采用李克特量表（1-5 分）对匹配质量评分  
- 考察模型在语言和文化理解方面的深度  

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 采用 LLM-as-judge 进行评估  
- 指标：平均李克特得分（1-5 分制）  
- 评估生成叙事的相关性、准确性、深度和细节  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `drivel_writing` |
| **数据集ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `Reasoning` |
| **指标** | `bert_score`, `gpt_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 600 |
| 提示词长度（平均） | 313.18 字符 |
| 提示词长度（最小/最大） | 256 / 717 字符 |

## 样例示例

**子集**: `narrative-writing-english`

```json
{
  "input": [
    {
      "id": "f47953a9",
      "content": [
        {
          "text": "You need to first read and understand the text given. Generate a detailed description to illustrate the implicit narrative of the text.\n\nPlease provide your response in English, with a clear and comprehensive explanation of the narrative.\n\nText: 後天的努力比什麼都重要，所以今天和明天休息。"
        }
      ]
    }
  ],
  "target": "This creates a paradoxical tone, as it acknowledges the value of diligence but simultaneously advocates for procrastination. The underlying message could reflect a lighthearted take on balancing work and rest or even poking fun at the tendency to delay responsibilities.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "text": "後天的努力比什麼都重要，所以今天和明天休息。",
    "reference_narrative": "This creates a paradoxical tone, as it acknowledges the value of diligence but simultaneously advocates for procrastination. The underlying message could reflect a lighthearted take on balancing work and rest or even poking fun at the tendency to delay responsibilities."
  }
}
```

## 提示模板

**提示模板：**
```text
You need to first read and understand the text given. Generate a detailed description to illustrate the implicit narrative of the text.

Please provide your response in English, with a clear and comprehensive explanation of the narrative.

Text: {text}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drivel_writing \
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
    datasets=['drivel_writing'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```