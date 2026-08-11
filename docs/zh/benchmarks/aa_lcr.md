# AA-LCR


## 概述

AA-LCR（Artificial Analysis Long Context Retrieval）是一个用于评估语言模型长上下文检索与推理能力的基准测试。该任务要求模型在多个文档中查找并综合信息以回答问题。

## 任务描述

- **任务类型**：长上下文问答（Long-Context Question Answering）
- **输入**：多个文档 + 需要跨文档推理的问题
- **输出**：从文档信息中综合得出的答案
- **上下文**：超长上下文（多个文档拼接而成）

## 核心特性

- 测试长上下文检索能力
- 多文档理解能力
- 要求跨文档推理
- 使用大语言模型（LLM）作为答案正确性评判器
- 自动下载文档语料库

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率**（通过 LLM 判定）
- 在 **test** 数据划分上进行评估
- 若未指定 `text_dir`，将自动下载文档
- 判定提示词会将候选答案与参考答案进行比较

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `aa_lcr` |
| **数据集ID** | [evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `LongContext`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 100 |
| 提示词长度（平均） | 414674.06 字符 |
| 提示词长度（最小/最大） | 240709 / 548771 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "0b26b81d",
      "content": "\nBEGIN INPUT DOCUMENTS\n\nBEGIN DOCUMENT 1:\n[Contents lists available at ScienceDirect](https://www.elsevier.com/locate/techfore)\n\n# Technological Forecasting & Social Change\n\n[journal homepage: www.elsevier.com/locate/techfore](http://www.else ... [TRUNCATED] ...  and undertakings issued by the ACCC. Identify and rank the industries explicitly mentioned in the paragraphs, according to the number of infringements over the past three decades. Exclude Broadcasting Industry from the answer.\n\nEND QUESTION\n"
    }
  ],
  "target": "1. Airline Industry (12)\\n2. Accommodation Industry (4)",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question": "Based on the provided documents, there appears to be a correlation between industry concentration and the frequency of consumer-related infringements and undertakings issued by the ACCC. Identify and rank the industries explicitly mentioned in the paragraphs, according to the number of infringements over the past three decades. Exclude Broadcasting Industry from the answer.",
    "data_source_urls": "https://competition-policy.ec.europa.eu/system/files/2024-06/A_taxonomy_of_industry_competition_launch.pdf;https://www.industry.gov.au/sites/default/files/2023-11/barriers-to-collaboration-and-commercialisation-iisa.pdf;https://e61.in/wp-content/uploads/2023/08/The-State-of-Competition.pdf;https://uu.diva-portal.org/smash/get/diva2:1798138/FULLTEXT01.pdf;https://one.oecd.org/document/DAF/COMP(2023)14/en/pdf",
    "input_tokens": 94494
  }
}
```

*注：部分内容为显示目的已被截断。*

## 提示模板

**提示模板：**
```text

BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION

```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `text_dir` | `str | null` | `None` | 包含已提取的 AA-LCR 文本文件的本地目录；若为 null，则自动下载并解压。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets aa_lcr \
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
    datasets=['aa_lcr'],
    dataset_args={
        'aa_lcr': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
