# GeniaNER

## 概述

GeniaNER 是一个大规模的生物医学命名实体识别（NER）数据集，包含 2,000 篇 MEDLINE 摘要，涵盖超过 40 万个单词和近 10 万个生物学术语标注。它是生物医学实体识别领域最全面的资源之一。

## 任务描述

- **任务类型**：生物医学命名实体识别（NER）
- **输入**：来自 GENIA 语料库的 MEDLINE 摘要
- **输出**：识别出的生物实体片段
- **领域**：分子生物学、生物信息学

## 主要特点

- 2,000 篇 MEDLINE 摘要
- 超过 40 万个单词
- 近 10 万个生物学术语标注
- 五种分子生物学实体类型
- 对生物分子实体的全面覆盖

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：CELL_LINE（细胞系）、CELL_TYPE（细胞类型）、DNA、PROTEIN（蛋白质）、RNA

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `genia_ner` |
| **数据集 ID** | [extraordinarylab/genia-ner](https://modelscope.cn/datasets/extraordinarylab/genia-ner/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估集** | `test` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,854 |
| 提示词长度（平均） | 3921.68 字符 |
| 提示词长度（最小/最大） | 3781 / 4433 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "3bb926a4",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nIL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase .\n\nOutput:\n<response><dna>IL-2 gene</dna> expression and <protein>NF ... [TRUNCATED] ...  a matching closing tag.\n\nText to process:\nThere is a single methionine codon-initiated open reading frame of 1,458 nt in frame with a homeobox and a CAX repeat , and the open reading frame is predicted to encode a protein of 51,659 daltons.\n"
    }
  ],
  "target": "<response>There is a single <dna>methionine codon-initiated open reading frame</dna> of 1,458 nt in frame with a <dna>homeobox</dna> and a <dna>CAX repeat</dna> , and the <dna>open reading frame</dna> is predicted to encode a protein of 51,659 daltons.</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "There",
      "is",
      "a",
      "single",
      "methionine",
      "codon-initiated",
      "open",
      "reading",
      "frame",
      "of",
      "1,458",
      "nt",
      "in",
      "frame",
      "with",
      "a",
      "homeobox",
      "and",
      "a",
      "CAX",
      "repeat",
      ",",
      "and",
      "the",
      "open",
      "reading",
      "frame",
      "is",
      "predicted",
      "to",
      "encode",
      "a",
      "protein",
      "of",
      "51,659",
      "daltons."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-DNA",
      "I-DNA",
      "I-DNA",
      "I-DNA",
      "I-DNA",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-DNA",
      "O",
      "O",
      "B-DNA",
      "I-DNA",
      "O",
      "O",
      "O",
      "B-DNA",
      "I-DNA",
      "I-DNA",
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
    --datasets genia_ner \
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
    datasets=['genia_ner'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```