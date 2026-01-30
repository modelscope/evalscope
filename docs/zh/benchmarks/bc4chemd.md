# BC4CHEMD


## 概述

BC4CHEMD（BioCreative IV CHEMDNER）数据集是一个包含 10,000 篇 PubMed 摘要的语料库，其中包含 84,355 个化学实体提及，均由专家手动标注，用于化学命名实体识别任务。

## 任务描述

- **任务类型**：化学命名实体识别（NER）
- **输入**：来自 PubMed 摘要的科学文本
- **输出**：识别出的化学化合物名称片段
- **领域**：化学、药理学、药物发现

## 主要特点

- 10,000 篇 PubMed 摘要
- 84,355 个化学实体提及
- 专家手动标注
- 来自 BioCreative IV 挑战赛的基准数据集
- 覆盖全面的化学化合物类型

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：CHEMICAL（化学物质和药物名称）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bc4chemd` |
| **数据集ID** | [extraordinarylab/bc4chemd](https://modelscope.cn/datasets/extraordinarylab/bc4chemd/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 26,364 |
| 提示词长度（平均） | 3042.58 字符 |
| 提示词长度（最小/最大） | 2879 / 3709 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "5067e19c",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nDPP6 as a candidate gene for neuroleptic - induced tardive dyskinesia .\n\nOutput:\n<response>DPP6 as a candidate gene for neuroleptic - induced tardive dyskinesia .</response>\n\nInput:\n ... [TRUNCATED] ... ry opening tag has a matching closing tag.\n\nText to process:\nEffects of docosahexaenoic acid and methylmercury on child ' s brain development due to consumption of fish by Finnish mother during pregnancy : a probabilistic modeling approach .\n"
    }
  ],
  "target": "<response>Effects of <chemical>docosahexaenoic acid</chemical> and <chemical>methylmercury</chemical> on child ' s brain development due to consumption of fish by Finnish mother during pregnancy : a probabilistic modeling approach .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Effects",
      "of",
      "docosahexaenoic",
      "acid",
      "and",
      "methylmercury",
      "on",
      "child",
      "'",
      "s",
      "brain",
      "development",
      "due",
      "to",
      "consumption",
      "of",
      "fish",
      "by",
      "Finnish",
      "mother",
      "during",
      "pregnancy",
      ":",
      "a",
      "probabilistic",
      "modeling",
      "approach",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "B-CHEMICAL",
      "I-CHEMICAL",
      "O",
      "B-CHEMICAL",
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
    --datasets bc4chemd \
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
    datasets=['bc4chemd'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```