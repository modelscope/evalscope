# AnatEM


## 概述

AnatEM 语料库是一个用于解剖学实体识别的丰富资源，通过扩展和整合先前的语料库构建而成。该语料库包含来自 1,212 篇生物医学文献的超过 13,000 个标注，专注于识别从亚细胞组分到器官系统的各类解剖结构。

## 任务描述

- **任务类型**：生物医学命名实体识别（NER）
- **输入**：来自 PubMed 摘要的生物医学文本
- **输出**：识别出的解剖学实体片段及其类型
- **领域**：解剖学、生物医学文献

## 主要特点

- 超过 13,000 个解剖学实体标注
- 来自 PubMed 的 1,212 篇生物医学文献
- 全面覆盖解剖结构（从细胞到器官）
- 由专家人工标注
- 适用于生物医学文本挖掘

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：ANATOMY（包括亚细胞结构、细胞、组织、器官）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `anat_em` |
| **数据集 ID** | [extraordinarylab/anat-em](https://modelscope.cn/datasets/extraordinarylab/anat-em/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,830 |
| 提示词长度（平均） | 3007.08 字符 |
| 提示词长度（最小/最大） | 2861 / 3652 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "64b20a23",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nImmunostaining and confocal analysis\n\nOutput:\n<response>Immunostaining and confocal analysis</response>\n\nInput:\nDNA labelling and staining with 5 - bromo - 2 ' - deoxyuridine ( BrdU  ... [TRUNCATED] ... Do not include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\n( a ) Schematic drawing of the magnetic tweezers .\n"
    }
  ],
  "target": "<response>( a ) Schematic drawing of the magnetic tweezers .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "(",
      "a",
      ")",
      "Schematic",
      "drawing",
      "of",
      "the",
      "magnetic",
      "tweezers",
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
    --datasets anat_em \
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
    datasets=['anat_em'],
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```