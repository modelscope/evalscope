# MathQA

## Overview

MathQA is a large-scale dataset for mathematical word problem solving, gathered by annotating the AQuA-RAT dataset with fully-specified operational programs using a new representation language. It contains diverse math problems requiring multi-step reasoning.

## Task Description

- **Task Type**: Mathematical Reasoning (Multiple-Choice)
- **Input**: Math word problem with multiple answer choices
- **Output**: Correct answer with chain-of-thought reasoning
- **Difficulty**: Varied (elementary to intermediate level)

## Key Features

- Annotated with executable operational programs
- Tests quantitative reasoning and problem-solving skills
- Diverse mathematical topics and question formats
- Multiple-choice format with structured solutions
- Useful for evaluating mathematical reasoning capabilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Evaluates on test split
- Simple accuracy metric
- Reasoning steps available in metadata

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `math_qa` |
| **Dataset ID** | [extraordinarylab/math-qa](https://modelscope.cn/datasets/extraordinarylab/math-qa/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,985 |
| Prompt Length (Mean) | 433.02 chars |
| Prompt Length (Min/Max) | 257 / 879 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "e77fca06",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E. Think step by step before answering.\n\na shopkeeper sold an article offering a discount of 5 % and earned a profit of 31.1 % . what would have been the percentage of profit earned if no discount had been offered ?\n\nA) 38\nB) 27.675\nC) 30\nD) data inadequate\nE) none of these"
    }
  ],
  "choices": [
    "38",
    "27.675",
    "30",
    "data inadequate",
    "none of these"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": "\"giving no discount to customer implies selling the product on printed price . suppose the cost price of the article is 100 . then printed price = 100 ã — ( 100 + 31.1 ) / ( 100 â ˆ ’ 5 ) = 138 hence , required % profit = 138 â € “ 100 = 38 % answer a\""
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
    --datasets math_qa \
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
    datasets=['math_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


