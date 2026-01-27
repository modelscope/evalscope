# GSM8K

GSM8K (Grade School Math 8K) is a dataset of 8.5K high-quality, linguistically diverse grade school math word problems created by human problem writers.

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
| **Train Split** | `train` |

## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,319 |
| Prompt Length (Mean) | 1966.87 chars |
| Prompt Length (Min/Max) | 1800 / 2575 chars |

## Sample Example

**Subset**: `main`

```json
{
  "input": [
    {
      "id": "3d025da2",
      "content": "Here are some examples of how to solve similar problems:\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nReasoning:\nNatalia sold 48/ ... [TRUNCATED] ... ds every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market."
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

<details>
<summary>Few-shot Template</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}

{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

</details>

## Usage

```python
from evalscope import run_task

results = run_task(
    model='your-model',
    datasets=['gsm8k'],
)
```