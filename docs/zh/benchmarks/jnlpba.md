# JNLPBA


## 概述

JNLPBA 数据集是一个广泛使用的生物实体识别资源，包含来自 GENIA 语料库的 2,404 篇 MEDLINE 摘要，标注了五种关键的分子生物学实体类型。它是生物医学命名实体识别（NER）的标准基准测试数据集。

## 任务描述

- **任务类型**：生物医学命名实体识别（NER）
- **输入**：来自 MEDLINE 摘要的生物医学文本
- **输出**：识别出的分子生物学实体片段
- **领域**：分子生物学、生物信息学

## 主要特点

- 包含 GENIA 语料库中的 2,404 篇 MEDLINE 摘要
- 涵盖五种分子生物学实体类型
- 由领域专家进行人工标注
- 是生物医学 NER 的标准基准
- 全面覆盖各类生物分子实体

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：PROTEIN（蛋白质）、DNA（脱氧核糖核酸）、RNA（核糖核酸）、CELL_LINE（细胞系）、CELL_TYPE（细胞类型）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `jnlpba` |
| **数据集ID** | [extraordinarylab/jnlpba](https://modelscope.cn/datasets/extraordinarylab/jnlpba/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,856 |
| 提示词长度（平均） | 3609.26 字符 |
| 提示词长度（最小/最大） | 3450 / 4664 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "270411f4",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nIL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase .\n\nOutput:\n<response><dna>IL-2 gene</dna> expression and <protein>NF ... [TRUNCATED] ... ged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nNumber of glucocorticoid receptors in lymphocytes and their sensitivity to hormone action .\n"
    }
  ],
  "target": "<response>Number of <protein>glucocorticoid receptors</protein> in <cell_type>lymphocytes</cell_type> and their sensitivity to hormone action .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Number",
      "of",
      "glucocorticoid",
      "receptors",
      "in",
      "lymphocytes",
      "and",
      "their",
      "sensitivity",
      "to",
      "hormone",
      "action",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "B-PROTEIN",
      "I-PROTEIN",
      "O",
      "B-CELL_TYPE",
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

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets jnlpba \
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
    datasets=['jnlpba'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```