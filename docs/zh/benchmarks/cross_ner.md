# CrossNER

## 概述

CrossNER 是一个完全标注的命名实体识别（NER）数据集，涵盖五个不同领域：人工智能（AI）、文学（Literature）、音乐（Music）、政治（Politics）和科学（Science）。该数据集支持跨领域 NER 评估和领域自适应研究。

## 任务描述

- **任务类型**：跨领域命名实体识别（NER）
- **输入**：来自五个专业领域的文本
- **输出**：领域特定的实体片段
- **领域**：AI、文学、音乐、政治、科学

## 主要特点

- 包含五个多样化的领域子集
- 每个子集具有领域特定的实体类型
- 支持跨领域迁移能力评估
- 由专家完全标注
- 适用于领域自适应研究

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 子集：ai、literature、music、politics、science
- 实体类型因领域子集而异

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cross_ner` |
| **数据集 ID** | [extraordinarylab/cross-ner](https://modelscope.cn/datasets/extraordinarylab/cross-ner/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认样本数** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,506 |
| 提示词长度（平均） | 5687.97 字符 |
| 提示词长度（最小/最大） | 5407 / 6007 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `ai` | 431 | 5562.3 | 5407 | 5878 |
| `literature` | 416 | 5725.13 | 5566 | 6007 |
| `music` | 465 | 5737.0 | 5570 | 5962 |
| `politics` | 651 | 5701.8 | 5527 | 5935 |
| `science` | 543 | 5700.71 | 5548 | 5995 |

## 样例示例

**子集**: `ai`

```json
{
  "input": [
    {
      "id": "3a78cbcf",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nPopular approaches of opinion-based recommender system utilize various techniques including text mining , information retrieval , sentiment analysis ( see also Multimodal sentiment a ... [TRUNCATED] ...  the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nTypical generative model approaches include naive Bayes classifier s , Gaussian mixture model s , variational autoencoders and others .\n"
    }
  ],
  "target": "<response>Typical generative model approaches include <algorithm>naive Bayes classifier</algorithm> s , <algorithm>Gaussian mixture model</algorithm> s , <algorithm>variational autoencoders</algorithm> and others .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Typical",
      "generative",
      "model",
      "approaches",
      "include",
      "naive",
      "Bayes",
      "classifier",
      "s",
      ",",
      "Gaussian",
      "mixture",
      "model",
      "s",
      ",",
      "variational",
      "autoencoders",
      "and",
      "others",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-ALGORITHM",
      "I-ALGORITHM",
      "I-ALGORITHM",
      "O",
      "O",
      "B-ALGORITHM",
      "I-ALGORITHM",
      "I-ALGORITHM",
      "O",
      "O",
      "B-ALGORITHM",
      "I-ALGORITHM",
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
    --datasets cross_ner \
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
    datasets=['cross_ner'],
    dataset_args={
        'cross_ner': {
            # subset_list: ['ai', 'literature', 'music']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```