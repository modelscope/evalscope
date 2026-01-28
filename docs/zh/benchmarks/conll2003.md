# CoNLL2003


## 概述

CoNLL-2003 是在 2003 年计算自然语言学习会议（Conference on Computational Natural Language Learning）上提出的经典命名实体识别（NER）基准数据集。该数据集包含标注了四种实体类型的新闻文章。

## 任务描述

- **任务类型**：命名实体识别（NER）
- **输入**：待识别实体的文本
- **输出**：带有类型标签的实体片段
- **实体类型**：人物（PER）、组织（ORG）、地点（LOC）、其他（MISC）

## 主要特点

- 标准 NER 基准，实体类型定义清晰
- 新闻领域文本，标注质量高
- 四类实体，定义明确
- 支持少样本（few-shot）评估
- 提供全面的评估指标（精确率、召回率、F1 分数、准确率）

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：**精确率（Precision）**、**召回率（Recall）**、**F1 分数（F1 Score）**、**准确率（Accuracy）**
- 训练集划分：**train**，评估集划分：**test**
- 实体类型映射为人类可读的名称


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `conll2003` |
| **数据集ID** | [extraordinarylab/conll2003](https://modelscope.cn/datasets/extraordinarylab/conll2003/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认样本数** | 5-shot |
| **评估集划分** | `test` |
| **训练集划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,453 |
| 提示词长度（平均） | 2733.98 字符 |
| 提示词长度（最小/最大） | 2663 / 3275 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "c8d2b130",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nEU rejects German call to boycott British lamb .\n\nOutput:\n<response><organization>EU</organization> rejects <miscellaneous>German</miscellaneous> call to boycott <miscellaneous>Briti ... [TRUNCATED] ... include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nSOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .\n"
    }
  ],
  "target": "<response>SOCCER - <location>JAPAN</location> GET LUCKY WIN , <person>CHINA</person> IN SURPRISE DEFEAT .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "SOCCER",
      "-",
      "JAPAN",
      "GET",
      "LUCKY",
      "WIN",
      ",",
      "CHINA",
      "IN",
      "SURPRISE",
      "DEFEAT",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "B-LOC",
      "O",
      "O",
      "O",
      "O",
      "B-PER",
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
<summary>少样本模板</summary>

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
    --datasets conll2003 \
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
    datasets=['conll2003'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```