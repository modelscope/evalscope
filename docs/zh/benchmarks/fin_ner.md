# FinNER


## 概述

FinNER 数据集是一个来自美国证券交易委员会（SEC）公开文件中的金融协议语料库，标注了人物（Person）、组织（Organization）、地点（Location）和杂项（Miscellaneous）实体，用于支持信用风险评估中的信息抽取任务。

## 任务描述

- **任务类型**：金融命名实体识别（NER）
- **输入**：来自 SEC 文件的金融协议文本
- **输出**：识别出的实体片段及其类型
- **领域**：金融、法律文档、信用风险

## 主要特点

- 来自 SEC 文件的金融协议
- 针对信用风险评估应用进行标注
- 采用适用于金融领域的标准 NER 实体类型
- 专为金融文档处理设计
- 适用于法律与金融领域的 AI 应用

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：PER（人物）、ORG（组织）、LOC（地点）、MISC（杂项）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `fin_ner` |
| **数据集 ID** | [extraordinarylab/fin-ner](https://modelscope.cn/datasets/extraordinarylab/fin-ner/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估集** | `test` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 305 |
| 提示词长度（平均） | 2891.13 字符 |
| 提示词长度（最小/最大） | 2663 / 6149 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "d8a8baf8",
      "content": "Here are some examples of named entity recognition:\n\nInput:\n( l ) \" Tranche A Shares \" has the meaning as defined in the Subscription Agreement .\n\nOutput:\n<response>( l ) \" Tranche A Shares \" has the meaning as defined in the Subscription Agr ... [TRUNCATED] ... osing tag.\n\nText to process:\nSubordinated Loan Agreement - Silicium de Provence SAS and Evergreen Solar Inc . 7 - December 2007 [ HERBERT SMITH LOGO ] ................................ 2007 SILICIUM DE PROVENCE SAS and EVERGREEN SOLAR , INC .\n"
    }
  ],
  "target": "<response>Subordinated Loan Agreement - <organization>Silicium de Provence SAS</organization> and <organization>Evergreen Solar Inc</organization> . 7 - December 2007 [ <person>HERBERT SMITH</person> LOGO ] ................................ 2007 <organization>SILICIUM DE PROVENCE SAS</organization> and <organization>EVERGREEN SOLAR</organization> , INC .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Subordinated",
      "Loan",
      "Agreement",
      "-",
      "Silicium",
      "de",
      "Provence",
      "SAS",
      "and",
      "Evergreen",
      "Solar",
      "Inc",
      ".",
      "7",
      "-",
      "December",
      "2007",
      "[",
      "HERBERT",
      "SMITH",
      "LOGO",
      "]",
      "................................",
      "2007",
      "SILICIUM",
      "DE",
      "PROVENCE",
      "SAS",
      "and",
      "EVERGREEN",
      "SOLAR",
      ",",
      "INC",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
      "O",
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-PER",
      "I-PER",
      "O",
      "O",
      "O",
      "O",
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
      "O",
      "B-ORG",
      "I-ORG",
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
    --datasets fin_ner \
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
    datasets=['fin_ner'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```