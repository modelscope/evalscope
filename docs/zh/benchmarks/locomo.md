# LoCoMo


## 概述

LoCoMo 用于评估两人在多轮会话对话中的超长期对话记忆能力。该适配器支持来自 `locomo10.json` 的官方问答任务。

## 任务描述

- **任务类型**：长上下文问答
- **输入**：包含会话日期的多轮对话历史和一个问题
- **输出**：简短的自由格式答案
- **子集**：`qa`

## 主要特性

- 使用托管在 ModelScope 上的官方 LoCoMo QA 数据文件
- 支持完整历史的长上下文提示和仅含证据的 oracle 提示
- 在数据中存在时包含图像说明，但不会下载图像文件
- 使用 LoCoMo 基于规则的 F1 / 对抗性拒绝评分机制，而非基于大语言模型（LLM）的评判器

## 评估说明

- 默认子集为 `qa`，且 `eval_mode=long_context`
- 使用 `extra_params.eval_mode='oracle_context'` 进行仅含证据的上限评估
- 此 QA 适配器不包含事件摘要、多模态对话生成和 RAG 检索任务

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `locomo` |
| **数据集ID** | [evalscope/locomo](https://modelscope.cn/datasets/evalscope/locomo/summary) |
| **论文** | N/A |
| **标签** | `LongContext`, `MultiTurn`, `QA` |
| **指标** | `f1` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,986 |
| 提示词长度（平均） | 94097.72 字符 |
| 提示词长度（最小/最大） | 55178 / 111026 字符 |

## 样例示例

**子集**: `qa`

```json
{
  "input": [
    {
      "id": "b585f7b0",
      "content": "Below is a conversation between two people: Caroline and Melanie. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\nDATE: 1:56 pm on 8 May, 2023\nCONVERSATION:\nC ... [TRUNCATED 74397 chars] ... m of a short phrase for the following question. Answer with exact words from the context whenever possible.\n\nQuestion: When did Caroline go to the LGBTQ support group? Use DATE of CONVERSATION to answer with an approximate date. Short answer:"
    }
  ],
  "target": "7 May 2023",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "sample_id": "conv-26",
    "qa_index": 0,
    "category": 2,
    "category_name": "temporal",
    "raw_question": "When did Caroline go to the LGBTQ support group?",
    "answer": "7 May 2023",
    "adversarial_answer": null,
    "eval_mode": "long_context",
    "question": "When did Caroline go to the LGBTQ support group? Use DATE of CONVERSATION to answer with an approximate date.",
    "evidence": [
      "D1:3"
    ]
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `long_context` | 评估模式：long_context 或 oracle_context。可选值：['long_context', 'oracle_context'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets locomo \
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
    datasets=['locomo'],
    dataset_args={
        'locomo': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```