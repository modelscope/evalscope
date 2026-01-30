# BC2GM


## 概述

BC2GM（BioCreative II Gene Mention）数据集是一个广泛使用的基因提及识别语料库，包含来自MEDLINE摘要的20,000个句子，其中的基因和蛋白质名称均由领域专家手动标注。

## 任务描述

- **任务类型**：生物医学命名实体识别（NER）
- **输入**：来自MEDLINE摘要的生物医学文本
- **输出**：识别出的基因和蛋白质名称片段
- **领域**：分子生物学、遗传学

## 主要特点

- 包含20,000个来自MEDLINE摘要的句子
- 由专家标注的基因和蛋白质提及
- 源自BioCreative II挑战赛的基准数据集
- 广泛用于生物医学NER评估
- 高质量的人工标注

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1分数（F1-Score）、准确率（Accuracy）
- 实体类型：GENE（基因和蛋白质名称）


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bc2gm` |
| **数据集ID** | [extraordinarylab/bc2gm](https://modelscope.cn/datasets/extraordinarylab/bc2gm/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,000 |
| 提示词长度（平均） | 2228.52 字符 |
| 提示词长度（最小/最大） | 2073 / 3129 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "06f75061",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nComparison with alkaline phosphatases and 5 - nucleotidase\n\nOutput:\n<response>Comparison with <gene>alkaline phosphatases</gene> and 5 - nucleotidase</response>\n\nInput:\nPharmacologic ... [TRUNCATED] ...  most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nPhenotypic analysis demonstrates that trio and Abl cooperate in regulating axon outgrowth in the embryonic central nervous system ( CNS ) .\n"
    }
  ],
  "target": "<response>Phenotypic analysis demonstrates that <gene>trio</gene> and <gene>Abl</gene> cooperate in regulating axon outgrowth in the embryonic central nervous system ( CNS ) .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Phenotypic",
      "analysis",
      "demonstrates",
      "that",
      "trio",
      "and",
      "Abl",
      "cooperate",
      "in",
      "regulating",
      "axon",
      "outgrowth",
      "in",
      "the",
      "embryonic",
      "central",
      "nervous",
      "system",
      "(",
      "CNS",
      ")",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-GENE",
      "O",
      "B-GENE",
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
    --datasets bc2gm \
    --limit 10  # 正式评估时请删除此行
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['bc2gm'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```