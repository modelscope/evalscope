# CoNLL++


## 概述

CoNLL++ 数据集是对广泛使用的 CoNLL2003 命名实体识别（NER）基准测试集中测试集的修正和清理版本。它提供了更高质量的标注，用于评估新闻文本上的命名实体识别系统。

## 任务描述

- **任务类型**：命名实体识别（NER）
- **输入**：新闻文章文本
- **输出**：带类型的已识别实体片段
- **领域**：新闻文章，通用领域

## 主要特点

- CoNLL2003 测试集的修正版本
- 标注质量高于原始版本
- 标准 NER 实体类型（PER、ORG、LOC、MISC）
- 广泛用于 NER 评估的基准测试
- 与原始 CoNLL2003 的结果具有可比性

## 评估说明

- 默认配置使用 **5-shot** 评估
- 指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：PER、ORG、LOC、MISC


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `conllpp` |
| **数据集ID** | [extraordinarylab/conllpp](https://modelscope.cn/datasets/extraordinarylab/conllpp/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,453 |
| 提示词长度（平均） | 2732.4 字符 |
| 提示词长度（最小/最大） | 2663 / 3155 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "d520596a",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nEU rejects German call to boycott British lamb .\n\nOutput:\n<response><organization>EU</organization> rejects <miscellaneous>German</miscellaneous> call to boycott <miscellaneous>Briti ... [TRUNCATED] ... include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nSOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .\n"
    }
  ],
  "target": "<response>SOCCER - <location>JAPAN</location> GET LUCKY WIN , <location>CHINA</location> IN SURPRISE DEFEAT .</response>",
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
      "B-LOC",
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
    --datasets conllpp \
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
    datasets=['conllpp'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```