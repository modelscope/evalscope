# BroadTwitterCorpus

## 概述

BroadTwitterCorpus 是一个在时间、地点和社会用途上分层采样的推文数据集。其目标是涵盖广泛的活动类型，从而提供一个更能代表 Twitter 这一最难处理的社交媒体语言形式的数据集。

## 任务描述

- **任务类型**：社交媒体命名实体识别（NER）
- **输入**：多样化的 Twitter 文本（推文）
- **输出**：识别出的实体片段及其类型
- **领域**：社交媒体，多样化上下文

## 主要特点

- 在时间、地点和用途上进行分层采样
- 能代表多样化的 Twitter 语言风格
- 针对社交媒体 NER 的挑战进行优化
- 包含三种标准 NER 实体类型（PER、ORG、LOC）
- 适用于鲁棒的社交媒体自然语言处理评估

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：PER、ORG、LOC

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `broad_twitter_corpus` |
| **数据集ID** | [extraordinarylab/broad-twitter-corpus](https://modelscope.cn/datasets/extraordinarylab/broad-twitter-corpus/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,000 |
| 提示词长度（平均） | 2341.71 字符 |
| 提示词长度（最小/最大） | 2246 / 2398 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "62394bfa",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nI hate the words chunder , vomit and puke . BUUH .\n\nOutput:\n<response>I hate the words chunder , vomit and puke . BUUH .</response>\n\nInput:\n♥ . . ) ) ( ♫ . ( ړײ ) ♫ . ♥ . « ▓ » ♥ . ♫ ... [TRUNCATED] ... planations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\n@ colgo hey , congrats to you and the team ! Always worth a read :)\n"
    }
  ],
  "target": "<response><person>@</person> <person>colgo</person> hey , congrats to you and the team ! Always worth a read :)</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "@",
      "colgo",
      "hey",
      ",",
      "congrats",
      "to",
      "you",
      "and",
      "the",
      "team",
      "!",
      "Always",
      "worth",
      "a",
      "read",
      ":)"
    ],
    "ner_tags": [
      "B-PER",
      "B-PER",
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
    --datasets broad_twitter_corpus \
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
    datasets=['broad_twitter_corpus'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```