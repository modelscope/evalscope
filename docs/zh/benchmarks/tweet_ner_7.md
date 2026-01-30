# TweetNER7


## 概述

TweetNER7 是一个大规模命名实体识别（NER）数据集，包含 2019 至 2021 年间超过 11,000 条推文，标注了七种实体类型，旨在促进对社交媒体语言短期时间变化的研究。

## 任务描述

- **任务类型**：社交媒体命名实体识别（NER）
- **输入**：2019–2021 年间的 Twitter 文本
- **输出**：识别出的七类实体片段
- **领域**：社交媒体、语言的时间变化

## 主要特点

- 超过 11,000 条已标注推文
- 七种多样化的实体类型
- 时间跨度为 2019–2021 年，支持时间维度分析
- 研究社交媒体中的语言演变
- 针对社交内容提供丰富的实体类型覆盖

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：CORPORATION（公司）、CREATIVE_WORK（创意作品）、EVENT（事件）、GROUP（群体）、LOCATION（地点）、PERSON（人物）、PRODUCT（产品）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tweet_ner_7` |
| **数据集 ID** | [extraordinarylab/tweet-ner-7](https://modelscope.cn/datasets/extraordinarylab/tweet-ner-7/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,383 |
| 提示词长度（平均） | 3364.87 字符 |
| 提示词长度（最小/最大） | 3226 / 3630 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "cde2f5cb",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nMorning 5km run with {{USERNAME}} for breast cancer awareness # pinkoctober # breastcancerawareness # zalorafit # zalorafitxbnwrc @ The Central Park , Desa Parkcity {{URL}}\n\nOutput:\n ... [TRUNCATED] ...  it 's been an amazing experience . Afro paper : the rolling stone project and I do n't think you know Popin boyz are projects that 's gonna shake the industry . Do n't sleep on {{USERNAME}} {{USERNAME}} {{USERNAME}} Watch out for these guys\n"
    }
  ],
  "target": "<response>Hanging out with the # <group>Popinboyz</group> and it 's been an amazing experience . <creative_work>Afro paper</creative_work> : <creative_work>the rolling stone project</creative_work> and I do n't think you know <group>Popin boyz</group> are projects that 's gonna shake the industry . Do n't sleep on {{USERNAME}} {{USERNAME}} {{USERNAME}} Watch out for these guys</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Hanging",
      "out",
      "with",
      "the",
      "#",
      "Popinboyz",
      "and",
      "it",
      "'s",
      "been",
      "an",
      "amazing",
      "experience",
      ".",
      "Afro",
      "paper",
      ":",
      "the",
      "rolling",
      "stone",
      "project",
      "and",
      "I",
      "do",
      "n't",
      "think",
      "you",
      "know",
      "Popin",
      "boyz",
      "are",
      "projects",
      "that",
      "'s",
      "gonna",
      "shake",
      "the",
      "industry",
      ".",
      "Do",
      "n't",
      "sleep",
      "on",
      "{{USERNAME}}",
      "{{USERNAME}}",
      "{{USERNAME}}",
      "Watch",
      "out",
      "for",
      "these",
      "guys"
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-GROUP",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "O",
      "B-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-GROUP",
      "I-GROUP",
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

*注：部分内容因展示需要已被截断。*

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
    --datasets tweet_ner_7 \
    --limit 10  # 正式评估时请移除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['tweet_ner_7'],
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```