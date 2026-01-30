# TweeBankNER


## 概述

Tweebank-NER 是一个英文 Twitter 语料库，通过对已进行句法分析的 Tweebank V2 数据集标注四种命名实体类型（人物、组织、地点和杂项）构建而成。该数据集旨在应对社交媒体非正式文本中的命名实体识别（NER）挑战。

## 任务描述

- **任务类型**：社交媒体命名实体识别（NER）
- **输入**：Twitter 文本（推文）
- **输出**：识别出的实体片段及其类型
- **领域**：社交媒体、非正式文本

## 主要特点

- 基于 Tweebank V2 的句法标注
- 包含四种标准 NER 实体类型
- 针对非正式语言带来的挑战
- 处理 Twitter 特有的文本特征（如话题标签、用户提及）
- 适用于社交媒体自然语言处理应用

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：PER（人物）、ORG（组织）、LOC（地点）、MISC（杂项）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tweebank_ner` |
| **数据集ID** | [extraordinarylab/tweebank-ner](https://modelscope.cn/datasets/extraordinarylab/tweebank-ner/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认样本数** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,201 |
| 提示词长度（平均） | 2322.67 字符 |
| 提示词长度（最小/最大） | 2250 / 2398 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "46ad4b5f",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nRT @USER2362 : Farmall Heart Of The Holidays Tabletop Christmas Tree With Lights And Motion URL1087 #Holiday #Gifts\n\nOutput:\n<response>RT @USER2362 : <organization>Farmall</organizat ... [TRUNCATED] ... include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\n@USER1812 No , I 'm not . It 's definitely not a rapper .\n"
    }
  ],
  "target": "<response>@USER1812 No , I 'm not . It 's definitely not a rapper .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "@USER1812",
      "No",
      ",",
      "I",
      "'m",
      "not",
      ".",
      "It",
      "'s",
      "definitely",
      "not",
      "a",
      "rapper",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
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

*注：部分内容为显示目的已被截断。*

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
    --datasets tweebank_ner \
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
    datasets=['tweebank_ner'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```