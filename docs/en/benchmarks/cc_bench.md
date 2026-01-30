# CCBench


## Overview

CCBench (Chinese Culture Bench) is an extension of MMBench specifically designed to evaluate multimodal models' understanding of Chinese traditional culture. It covers various aspects of Chinese cultural heritage through visual question answering.

## Task Description

- **Task Type**: Visual Multiple-Choice Q&A (Chinese Culture)
- **Input**: Image with question about Chinese culture
- **Output**: Single correct answer letter (A, B, C, or D)
- **Language**: Primarily Chinese content

## Key Features

- Questions about Chinese traditional culture
- Categories: Calligraphy, Painting, Cultural Relics, Food & Clothes
- Historical Figures, Scenery & Building, Sketch Reasoning, Traditional Shows
- Tests cultural knowledge combined with visual understanding
- Extension of the MMBench evaluation framework

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting
- Evaluates on test split
- Simple accuracy metric for scoring
- Requires both visual perception and cultural knowledge


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cc_bench` |
| **Dataset ID** | [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,040 |
| Prompt Length (Mean) | 270.1 chars |
| Prompt Length (Min/Max) | 254 / 394 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,040 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 119x118 - 512x512 |
| Formats | jpeg |


## Sample Example

**Subset**: `cc`

```json
{
  "input": [
    {
      "id": "2797f551",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\n图中所示建筑名称为？\n\nA) 天坛\nB) 故宫\nC) 黄鹤楼\nD) 少林寺"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~22.7KB]"
        }
      ]
    }
  ],
  "choices": [
    "天坛",
    "故宫",
    "黄鹤楼",
    "少林寺"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "category": "scenery_building",
    "source": "https://zh.wikipedia.org/wiki/%E5%A4%A9%E5%9D%9B"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cc_bench \
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
    datasets=['cc_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


