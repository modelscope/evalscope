# MultiNERD

## 概述

MultiNERD 是一个大规模、多语言、多体裁的细粒度命名实体识别（Named Entity Recognition, NER）数据集，通过维基百科（Wikipedia）和维基新闻（Wikinews）自动生成。该数据集涵盖 10 种语言和 15 个不同的实体类别。

## 任务描述

- **任务类型**：细粒度多语言命名实体识别（NER）
- **输入**：维基百科和维基新闻文本
- **输出**：识别出的实体片段及其对应的 15 种细粒度类型
- **领域**：通用知识、新闻、百科类内容

## 主要特性

- 大规模自动构建的语料库
- 支持 10 种语言
- 包含 15 种细粒度实体类别
- 数据来源为维基百科和维基新闻
- 实体类型覆盖全面

## 评估说明

- 默认配置使用 **5-shot** 评估方式
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型包括：PER（人物）、ORG（组织）、LOC（地点）、ANIM（动物）、BIO（生物）、CEL（细胞）、DIS（疾病）、EVE（事件）、FOOD（食物）、INST（机构）、MEDIA（媒体）、MYTH（神话）、PLANT（植物）、TIME（时间）、VEHI（交通工具）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `multi_nerd` |
| **数据集ID** | [extraordinarylab/multi-nerd](https://modelscope.cn/datasets/extraordinarylab/multi-nerd/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 167,993 |
| 提示词长度（平均） | 4016.26 字符 |
| 提示词长度（最小/最大） | 3915 / 4501 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "56cf0758",
      "content": "Here are some examples of named entity recognition:\n\nInput:\n2002 ging er ins Ausland und wechselte für 750.000 Pfund Sterling zu Manchester City .\n\nOutput:\n<response>2002 ging er ins Ausland und wechselte für 750.000 Pfund Sterling zu <organi ... [TRUNCATED] ...  Ensure every opening tag has a matching closing tag.\n\nText to process:\nIn der Wissenschaft und dort vor allem in der Soziologie wird der Begriff Lebensführung traditionell stark mit der religionshistorischen Arbeit von Max Weber verbunden .\n"
    }
  ],
  "target": "<response>In der Wissenschaft und dort vor allem in der Soziologie wird der Begriff Lebensführung traditionell stark mit der religionshistorischen Arbeit von <person>Max Weber</person> verbunden .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "In",
      "der",
      "Wissenschaft",
      "und",
      "dort",
      "vor",
      "allem",
      "in",
      "der",
      "Soziologie",
      "wird",
      "der",
      "Begriff",
      "Lebensführung",
      "traditionell",
      "stark",
      "mit",
      "der",
      "religionshistorischen",
      "Arbeit",
      "von",
      "Max",
      "Weber",
      "verbunden",
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
      "B-PER",
      "I-PER",
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
    --datasets multi_nerd \
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
    datasets=['multi_nerd'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```