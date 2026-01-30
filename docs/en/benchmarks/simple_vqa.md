# SimpleVQA


## Overview

SimpleVQA is the first comprehensive multimodal benchmark to evaluate the factuality ability of MLLMs to answer natural language short questions. It features high-quality, challenging queries with static and timeless reference answers.

## Task Description

- **Task Type**: Factual Visual Question Answering
- **Input**: Image + factual question
- **Output**: Short factual answer
- **Domains**: Factuality, visual reasoning, knowledge recall

## Key Features

- Covers multiple tasks and scenarios
- High-quality, challenging questions
- Static and timeless reference answers (no temporal dependencies)
- Straightforward evaluation methodology
- Tests genuine factual knowledge beyond pattern matching

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with LLM judge
- Three-grade evaluation: CORRECT, INCORRECT, NOT_ATTEMPTED
- LLM judge uses detailed grading rubric for semantic matching
- Rich metadata includes language, source, and atomic facts


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `simple_vqa` |
| **Dataset ID** | [m-a-p/SimpleVQA](https://modelscope.cn/datasets/m-a-p/SimpleVQA/summary) |
| **Paper** | N/A |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,025 |
| Prompt Length (Mean) | 56.22 chars |
| Prompt Length (Min/Max) | 27 / 1015 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,025 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 106x56 - 5119x3413 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "4340fc24",
      "content": [
        {
          "text": "Answer the question:\n\n图中所示穴位所属的经脉是什么？"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~26.5KB]"
        }
      ]
    }
  ],
  "target": "足阳明胃经",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "data_id": 0,
    "image_description": "",
    "language": "CN",
    "original_category": "中华文化_中医",
    "source": "https://baike.baidu.com/item/%E4%BC%8F%E5%85%94%E7%A9%B4/3503684#:~:text\\u003d%E4%BA%BA%E4%BD%93%E7%A9%B4%E4%BD%8D%E5%90%8D%E4%BC%8F%E5%85%94%E7%A9%B4F%C3%BA%20t%C3%B9%EF%BC%88ST32%EF%BC%89%E5%B1%9E%E8%B6%B3%E9%98%B3%E6%98%8E%E8%83%83%E7%BB%8 ... [TRUNCATED] ... 4%BE%A7%E7%AB%AF%E7%9A%84%E8%BF%9E%E7%BA%BF%E4%B8%8A%EF%BC%8C%E9%AB%8C%E9%AA%A8%E4%B8%8A%E7%BC%98%E4%B8%8A6%E5%AF%B8%E3%80%82%E4%BC%8F%E5%85%94%E5%88%AB%E5%90%8D%E5%A4%96%E4%B8%98%E3%80%81%E5%A4%96%E5%8B%BE%EF%BC%8C%E4%BD%8D%E4%BA%8E%E5%A4%A7",
    "atomic_question": "图中所示穴位的名称是什么？",
    "atomic_fact": "伏兔"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the question:

{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets simple_vqa \
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
    datasets=['simple_vqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


