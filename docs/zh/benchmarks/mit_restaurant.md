# MIT-Restaurant

## 概述

MIT-Restaurant 数据集是一组专门用于训练和测试命名实体识别（NER）自然语言处理（NLP）模型的餐厅评论文本。该数据集包含来自真实评论的句子，并采用 BIO 格式进行标注。

## 任务描述

- **任务类型**：餐厅领域命名实体识别（NER）
- **输入**：餐厅评论文本和查询
- **输出**：识别出的与餐厅相关的实体片段
- **领域**：餐饮服务、餐厅评论、对话系统

## 主要特点

- 真实的餐厅评论句子
- 采用 BIO 格式标注
- 包含八种餐厅特定的实体类型
- 适用于餐饮服务领域的 NLP 任务
- 适配于对话式 AI 应用

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：AMENITY（设施）、CUISINE（菜系）、DISH（菜品）、HOURS（营业时间）、LOCATION（位置）、PRICE（价格）、RATING（评分）、RESTAURANT_NAME（餐厅名称）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mit_restaurant` |
| **数据集ID** | [extraordinarylab/mit-restaurant](https://modelscope.cn/datasets/extraordinarylab/mit-restaurant/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,521 |
| 提示词长度（平均） | 2383.97 字符 |
| 提示词长度（最小/最大） | 2338 / 2474 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "9d5a77f1",
      "content": "Here are some examples of named entity recognition:\n\nInput:\ncan you find me the cheapest mexican restaurant nearby\n\nOutput:\n<response>can you find me the <price>cheapest</price> <cuisine>mexican</cuisine> restaurant <location>nearby</location ... [TRUNCATED] ... mes provided.\n5. Do not include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\na four star restaurant with a bar\n"
    }
  ],
  "target": "<response>a <rating>four star</rating> restaurant <location>with a</location> <amenity>bar</amenity></response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "a",
      "four",
      "star",
      "restaurant",
      "with",
      "a",
      "bar"
    ],
    "ner_tags": [
      "O",
      "B-RATING",
      "I-RATING",
      "O",
      "B-LOCATION",
      "I-LOCATION",
      "B-AMENITY"
    ]
  }
}
```

*注：部分内容因显示需要已被截断。*

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
    --datasets mit_restaurant \
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
    datasets=['mit_restaurant'],
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```