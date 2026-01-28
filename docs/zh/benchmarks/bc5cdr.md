# BC5CDR


## 概述

BC5CDR 语料库是一个为 BioCreative V 挑战赛开发的手动标注资源，包含 1,500 篇 PubMed 文章，其中包含超过 4,400 个化学物质提及、5,800 个疾病提及以及 3,100 个化学-疾病相互作用。

## 任务描述

- **任务类型**：生物医学命名实体识别（NER）
- **输入**：PubMed 文章文本
- **输出**：识别出的化学物质和疾病实体范围
- **领域**：药理学、医学信息学、毒理学

## 主要特点

- 1,500 篇带有专家标注的 PubMed 文章
- 4,400+ 个化学物质提及
- 5,800+ 个疾病提及
- 3,100+ 个化学-疾病相互作用
- 来自 BioCreative V 挑战赛的基准数据集

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：CHEMICAL（化学物质）、DISEASE（疾病）


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bc5cdr` |
| **数据集ID** | [extraordinarylab/bc5cdr](https://modelscope.cn/datasets/extraordinarylab/bc5cdr/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估集** | `test` |
| **训练集** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,797 |
| 提示词长度（平均） | 3598.18 字符 |
| 提示词长度（最小/最大） | 3455 / 4087 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "154a621d",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nSelegiline - induced postural hypotension in Parkinson ' s disease : a longitudinal study on the effects of drug withdrawal .\n\nOutput:\n<response><chemical>Selegiline</chemical> - ind ... [TRUNCATED] ... e.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nTorsade de pointes ventricular tachycardia during low dose intermittent dobutamine treatment in a patient with dilated cardiomyopathy and congestive heart failure .\n"
    }
  ],
  "target": "<response><disease>Torsade de pointes ventricular tachycardia</disease> during low dose intermittent <chemical>dobutamine</chemical> treatment in a patient with <disease>dilated cardiomyopathy</disease> and <disease>congestive heart failure</disease> .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Torsade",
      "de",
      "pointes",
      "ventricular",
      "tachycardia",
      "during",
      "low",
      "dose",
      "intermittent",
      "dobutamine",
      "treatment",
      "in",
      "a",
      "patient",
      "with",
      "dilated",
      "cardiomyopathy",
      "and",
      "congestive",
      "heart",
      "failure",
      "."
    ],
    "ner_tags": [
      "B-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "O",
      "O",
      "O",
      "O",
      "B-CHEMICAL",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-DISEASE",
      "I-DISEASE",
      "O",
      "B-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
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

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bc5cdr \
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
    datasets=['bc5cdr'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```