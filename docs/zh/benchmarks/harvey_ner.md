# HarveyNER

## 概述

HarveyNER 是一个在飓风哈维（Hurricane Harvey）期间收集的推文数据集，其中标注了细粒度的位置信息。该数据集在非正式的危机相关描述中包含复杂且较长的位置提及，带来了独特的挑战。

## 任务描述

- **任务类型**：危机领域位置命名实体识别（NER）
- **输入**：与飓风哈维相关的推文
- **输出**：细粒度的位置实体片段
- **领域**：危机通信、灾害响应

## 主要特点

- 推文中包含细粒度的位置标注
- 位置提及复杂且长度较长
- 非正式的危机相关文本
- 四种位置实体类型（AREA、POINT、RIVER、ROAD）
- 适用于灾害响应相关的自然语言处理应用

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：AREA、POINT、RIVER、ROAD

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `harvey_ner` |
| **数据集ID** | [extraordinarylab/harvey-ner](https://modelscope.cn/datasets/extraordinarylab/harvey-ner/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,303 |
| 提示词长度（平均） | 3018.97 字符 |
| 提示词长度（最小/最大） | 2909 / 3199 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "629de70e",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nJust received word that UHVictoria and Bayou Oaks residents are in need of blankets , pillows , and clothes ( men & amp ; women ) . ( PT1 )\n\nOutput:\n<response>Just received word that ... [TRUNCATED] ... lap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nBREAKING : One firefighter injured after a fire / apparent explosion at the Lone Star Legal Aid Services in Downtown Houston\n"
    }
  ],
  "target": "<response>BREAKING : One firefighter injured after a fire / apparent explosion at the <point>Lone Star Legal Aid Services in Downtown Houston</point></response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "BREAKING",
      ":",
      "One",
      "firefighter",
      "injured",
      "after",
      "a",
      "fire",
      "/",
      "apparent",
      "explosion",
      "at",
      "the",
      "Lone",
      "Star",
      "Legal",
      "Aid",
      "Services",
      "in",
      "Downtown",
      "Houston"
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
      "O",
      "O",
      "O",
      "O",
      "B-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT"
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
    --datasets harvey_ner \
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
    datasets=['harvey_ner'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```