# Chinese-SimpleQA


## Overview

Chinese SimpleQA is a Chinese question-answering dataset designed to evaluate the performance of language models on simple factual questions. It tests the model's ability to understand and generate correct answers in Chinese across various knowledge domains.

## Task Description

- **Task Type**: Chinese Factual Question Answering
- **Input**: Simple factual question in Chinese
- **Output**: Factual answer in Chinese
- **Language**: Chinese

## Key Features

- Diverse topics covering various knowledge domains
- Simple factual questions testing world knowledge
- Chinese language evaluation
- LLM-as-judge evaluation for answer correctness
- Multiple category subsets available

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM-as-judge for evaluation
- Metrics: is_correct, is_incorrect, is_not_attempted
- Evaluates factual accuracy without requiring exact match


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `chinese_simpleqa` |
| **Dataset ID** | [AI-ModelScope/Chinese-SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary) |
| **Paper** | N/A |
| **Tags** | `Chinese`, `Knowledge`, `QA` |
| **Metrics** | `is_correct`, `is_incorrect`, `is_not_attempted` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,000 |
| Prompt Length (Mean) | 32.45 chars |
| Prompt Length (Min/Max) | 16 / 129 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `中华文化` | 326 | 32.09 | 18 | 86 |
| `人文与社会科学` | 609 | 33.94 | 18 | 87 |
| `工程、技术与应用科学` | 481 | 33.13 | 18 | 91 |
| `生活、艺术与文化` | 601 | 32.4 | 17 | 76 |
| `社会` | 453 | 32.33 | 18 | 129 |
| `自然与自然科学` | 530 | 30.49 | 16 | 83 |

## Sample Example

**Subset**: `中华文化`

```json
{
  "input": [
    {
      "id": "b6b48177",
      "content": "请回答问题：\n\n伏兔穴所属的经脉是什么？"
    }
  ],
  "target": "足阳明胃经",
  "id": 0,
  "group_id": 0,
  "subset_key": "中华文化",
  "metadata": {
    "id": "97e7f58a3b154facaa3a5c64d678c7bf",
    "primary_category": "中华文化",
    "secondary_category": "中医"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
请回答问题：

{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets chinese_simpleqa \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['chinese_simpleqa'],
    dataset_args={
        'chinese_simpleqa': {
            # subset_list: ['中华文化', '人文与社会科学', '工程、技术与应用科学']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


