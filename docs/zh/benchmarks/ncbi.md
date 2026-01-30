# NCBI


## 概述

NCBI 疾病语料库是一个人工标注的 PubMed 摘要资源，专为疾病名称识别与标准化而设计。它为评估疾病命名实体识别（NER）系统提供了黄金标准。

## 任务描述

- **任务类型**：疾病命名实体识别（Disease Named Entity Recognition, NER）
- **输入**：PubMed 摘要文本
- **输出**：识别出的疾病实体范围（spans）
- **领域**：医学信息学、临床自然语言处理（Clinical NLP）

## 主要特性

- 人工标注的疾病提及
- 基于 PubMed 摘要的语料库
- 支持疾病名称标准化
- 疾病 NER 评估的黄金标准
- 高质量的专家标注

## 评估说明

- 默认配置使用 **5-shot** 评估
- 评估指标：精确率（Precision）、召回率（Recall）、F1 分数（F1-Score）、准确率（Accuracy）
- 实体类型：DISEASE（包括疾病、障碍、综合征等）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ncbi` |
| **数据集 ID** | [extraordinarylab/ncbi](https://modelscope.cn/datasets/extraordinarylab/ncbi/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `NER` |
| **指标** | `precision`, `recall`, `f1_score`, `accuracy` |
| **默认示例数量** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 940 |
| 提示词长度（平均） | 2646.25 字符 |
| 提示词长度（最小/最大） | 2502 / 2988 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "e87ce89c",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nIdentification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor .\n\nOutput:\n<response>Identification of APC2 , a homologue of the <disease>adenomatous polypos ... [TRUNCATED] ...  If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nClustering of missense mutations in the ataxia - telangiectasia gene in a sporadic T - cell leukaemia .\n"
    }
  ],
  "target": "<response>Clustering of missense mutations in the <disease>ataxia - telangiectasia</disease> gene in a <disease>sporadic T - cell leukaemia</disease> .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Clustering",
      "of",
      "missense",
      "mutations",
      "in",
      "the",
      "ataxia",
      "-",
      "telangiectasia",
      "gene",
      "in",
      "a",
      "sporadic",
      "T",
      "-",
      "cell",
      "leukaemia",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "O",
      "O",
      "O",
      "B-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
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

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ncbi \
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
    datasets=['ncbi'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```