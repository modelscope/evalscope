# OntoNotes5

## 概述

OntoNotes Release 5.0 是一个大规模、多语言语料库，包含英语、中文和阿拉伯语文本，涵盖多种体裁。该语料库包含丰富的多层次语言学标注信息，包括句法、谓词-论元结构、词义、命名实体和共指关系。

## 任务描述

- **任务类型**：多体裁命名实体识别（NER）
- **输入**：来自新闻、博客、广播对话的文本
- **输出**：细粒度命名实体片段
- **语言**：英语、中文、阿拉伯语

## 主要特点

- 大规模多语言语料库
- 多种体裁（新闻、博客、广播）
- 18 种细粒度实体类型
- 丰富的语言学标注
- 命名实体识别评估的标准基准

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：PERSON（人物）、NORP（民族/宗教/政治团体）、FAC（设施）、ORG（组织）、GPE（地缘政治实体）、LOC（地点）、PRODUCT（产品）、EVENT（事件）、WORK_OF_ART（艺术作品）、LAW（法律）、LANGUAGE（语言）、DATE（日期）、TIME（时间）、PERCENT（百分比）、MONEY（货币）、QUANTITY（数量）、ORDINAL（序数）、CARDINAL（基数）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ontonotes5` |
| **数据集ID** | [extraordinarylab/ontonotes5](https://modelscope.cn/datasets/extraordinarylab/ontonotes5/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 8,262 |
| 提示词长度（平均） | 3364.28 字符 |
| 提示词长度（最小/最大） | 3253 / 4171 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "6215273c",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nPeople start their own businesses for many reasons .\n\nOutput:\n<response>People start their own businesses for many reasons .</response>\n\nInput:\nBut a chance to fill out sales - tax r ... [TRUNCATED] ... ening tag has a matching closing tag.\n\nText to process:\nThe following were among Friday 's offerings and pricings in the U.S. and non-U.S. capital markets , with terms and syndicate manager , as compiled by Dow Jones Capital Markets Report :\n"
    }
  ],
  "target": "<response>The following were among <date>Friday</date> 's offerings and pricings in the <geopolitical_entity>U.S.</geopolitical_entity> and <geopolitical_entity>non-U.S.</geopolitical_entity> capital markets , with terms and syndicate manager , as compiled by <organization>Dow Jones Capital Markets Report</organization> :</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "The",
      "following",
      "were",
      "among",
      "Friday",
      "'s",
      "offerings",
      "and",
      "pricings",
      "in",
      "the",
      "U.S.",
      "and",
      "non-U.S.",
      "capital",
      "markets",
      ",",
      "with",
      "terms",
      "and",
      "syndicate",
      "manager",
      ",",
      "as",
      "compiled",
      "by",
      "Dow",
      "Jones",
      "Capital",
      "Markets",
      "Report",
      ":"
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-DATE",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-GPE",
      "O",
      "B-GPE",
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
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
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
<summary>少样本提示模板</summary>

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

### 使用命令行接口（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ontonotes5 \
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
    datasets=['ontonotes5'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```