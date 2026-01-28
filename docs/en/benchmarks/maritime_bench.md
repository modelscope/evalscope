# MaritimeBench

## Overview

MaritimeBench is a benchmark for evaluating AI models on maritime-related multiple-choice questions in Chinese. It consists of specialized questions related to maritime knowledge, navigation, marine engineering, and seafaring operations.

## Task Description

- **Task Type**: Maritime Knowledge Multiple-Choice QA (Chinese)
- **Input**: Maritime-related question with 4 answer choices (A-D)
- **Output**: Correct answer letter in brackets [A/B/C/D]
- **Language**: Chinese

## Key Features

- Specialized maritime domain questions
- Chinese language evaluation
- Multiple subsets covering different maritime topics
- Tests professional maritime knowledge
- Standardized Chinese exam format

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on test split
- Simple accuracy metric
- Answers extracted using regex for bracketed format [A/B/C/D]

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `maritime_bench` |
| **Dataset ID** | [HiDolphin/MaritimeBench](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary) |
| **Paper** | N/A |
| **Tags** | `Chinese`, `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,888 |
| Prompt Length (Mean) | 259.04 chars |
| Prompt Length (Min/Max) | 173 / 717 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "0b8c5f63",
      "content": "请回答单选题。要求只输出选项，不输出解释，将选项放在[]里，直接输出答案。示例：\n\n题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。\n选项：\nA. 电磁力\nB. 压拉应力\nC. 弯曲应力\nD. 扭应力\n答：[A]\n 当前题目\n 船舶电力系统供电网络中，放射形网络的特点是______。①发散形传输②环形传输③缺乏冗余④冗余性能好\n选项：\nA. ②③\nB. ①③\nC. ②④\nD. ①④"
    }
  ],
  "choices": [
    "②③",
    "①③",
    "②④",
    "①④"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0
}
```

## Prompt Template

**Prompt Template:**
```text
请回答单选题。要求只输出选项，不输出解释，将选项放在[]里，直接输出答案。示例：

题目：在船舶主推进动力装置中，传动轴系在运转中承受以下复杂的应力和负荷，但不包括______。
选项：
A. 电磁力
B. 压拉应力
C. 弯曲应力
D. 扭应力
答：[A]
 当前题目
 {question}
选项：
{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets maritime_bench \
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
    datasets=['maritime_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


