# MIT-Movie-Trivia

## 概述

MIT-Movie-Trivia 数据集最初是为电影领域对话中的槽位填充任务而创建的，后经合并和过滤槽位类型，被改造用于命名实体识别（NER）。该数据集用于测试在对话式查询中识别电影相关实体的能力。

## 任务描述

- **任务类型**：电影领域命名实体识别（NER）
- **输入**：与电影相关的对话式查询
- **输出**：识别出的电影实体片段
- **领域**：娱乐、电影冷知识、对话系统

## 主要特点

- 电影领域的对话式查询
- 覆盖丰富的电影实体类型
- 由槽位填充任务改编而来
- 包含十二种涵盖电影属性的实体类型
- 适用于娱乐领域的自然语言处理任务

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：ACTOR（演员）、AWARD（奖项）、CHARACTER_NAME（角色名）、DIRECTOR（导演）、GENRE（类型）、OPINION（观点）、ORIGIN（起源地）、PLOT（剧情）、QUOTE（台词）、RELATIONSHIP（关系）、SOUNDTRACK（配乐）、YEAR（年份）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mit_movie_trivia` |
| **数据集ID** | [extraordinarylab/mit-movie-trivia](https://modelscope.cn/datasets/extraordinarylab/mit-movie-trivia/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估集** | `test` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,953 |
| 提示词长度（平均） | 3309.63 字符 |
| 提示词长度（最小/最大） | 3216 / 3565 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "c33a15eb",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nwhat 1995 romantic comedy film starred michael douglas as a u s head of state looking for love\n\nOutput:\n<response>what <year>1995</year> <genre>romantic comedy</genre> film starred < ... [TRUNCATED] ... If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\ni need that movie which involves aliens invading earth in a particular united states place in california\n"
    }
  ],
  "target": "<response>i need that movie which involves <plot>aliens invading earth in a particular united states place in california</plot></response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "i",
      "need",
      "that",
      "movie",
      "which",
      "involves",
      "aliens",
      "invading",
      "earth",
      "in",
      "a",
      "particular",
      "united",
      "states",
      "place",
      "in",
      "california"
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT"
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
    --datasets mit_movie_trivia \
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
    datasets=['mit_movie_trivia'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```