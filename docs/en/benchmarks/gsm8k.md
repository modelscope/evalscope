# GSM8K

GSM8K (Grade School Math 8K) is a dataset of 8.5K high-quality, linguistically diverse grade school math word problems created by human problem writers.

**Task Type**: Mathematical Reasoning
**Difficulty**: Grade School Level
**Answer Format**: Numerical answer enclosed in \boxed{{}}

The dataset is designed to support the task of question answering on basic mathematical problems that require multi-step reasoning. Each problem consists of a natural language question and a numerical answer. The problems test basic arithmetic operations and require 2-8 steps to solve.

**Key Features**:
- High-quality human-written problems
- Linguistically diverse question formulations
- Multi-step reasoning required
- Clear numerical answers

## Overview

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gsm8k` |
| **Dataset ID** | [AI-ModelScope/gsm8k](https://modelscope.cn/datasets/AI-ModelScope/gsm8k/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2110.14168) |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 4-shot |
| **Evaluation Split** | `test` |

## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,319 |
| Prompt Length (Mean) | 1966.87 chars |
| Prompt Length (Min/Max) | 1800 / 2575 chars |

## Subsets

- `main` (1,319 samples)

## Sample Example

**Subset**: `main`

```json
{
  "input": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market."
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## Usage

```python
from evalscope import run_task

results = run_task(
    model='your-model',
    datasets=['gsm8k'],
)
```