# JNLPBA-Rare

## 概述

JNLPBA-Rare 数据集是 JNLPBA 测试集的一个专门子集，用于评估模型在最不常见的实体类型（RNA 和细胞系）上的零样本（zero-shot）性能。该数据集用于测试模型识别稀有生物医学实体的能力。

## 任务描述

- **任务类型**：稀有生物医学命名实体识别（NER）
- **输入**：来自 MEDLINE 摘要的生物医学文本
- **输出**：识别出的 RNA 和细胞系实体范围
- **领域**：分子生物学、生物信息学

## 主要特点

- 聚焦于稀有实体类型（RNA、细胞系）
- 作为 JNLPBA 的子集，专用于零样本评估
- 测试模型对低频生物医学实体的处理能力
- 是命名实体识别任务中具有挑战性的基准
- 适用于评估模型在长尾分布上的表现

## 评估说明

- 默认配置使用 **0-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：RNA、CELL_LINE

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `jnlpba_rare` |
| **数据集ID** | [extraordinarylab/jnlpba-rare](https://modelscope.cn/datasets/extraordinarylab/jnlpba-rare/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认样本数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 465 |
| 提示词长度（平均） | 1111.66 字符 |
| 提示词长度（最小/最大） | 976 / 1403 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "11496ce5",
      "content": "You are a named entity recognition system that identifies the following entity types:\nrna (Names of RNA molecules), cell_line (Names of specific, cultured cell lines)\n\nProcess the provided text and mark all named entities with XML-style tags. ... [TRUNCATED] ... rlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nOctamer-binding proteins from B or HeLa cells stimulate transcription of the immunoglobulin heavy-chain promoter in vitro .\n"
    }
  ],
  "target": "<response>Octamer-binding proteins from <cell_line>B or HeLa cells</cell_line> stimulate transcription of the immunoglobulin heavy-chain promoter in vitro .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Octamer-binding",
      "proteins",
      "from",
      "B",
      "or",
      "HeLa",
      "cells",
      "stimulate",
      "transcription",
      "of",
      "the",
      "immunoglobulin",
      "heavy-chain",
      "promoter",
      "in",
      "vitro",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "B-CELL_LINE",
      "I-CELL_LINE",
      "I-CELL_LINE",
      "I-CELL_LINE",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O"
    ]
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

<details>
<summary>少样本（Few-shot）模板</summary>

```text
Here are some examples of named entity recognition:

{fewshot}

You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

</details>

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets jnlpba_rare \
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
    datasets=['jnlpba_rare'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```