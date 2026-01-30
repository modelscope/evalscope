# MathVista


## Overview

MathVista is a comprehensive benchmark for mathematical reasoning in visual contexts. It combines newly created datasets with existing benchmarks to evaluate models on diverse visual mathematical reasoning tasks across multiple domains.

## Task Description

- **Task Type**: Visual Mathematical Reasoning
- **Input**: Image with mathematical question (multiple-choice or free-form)
- **Output**: Numerical answer or answer choice
- **Domains**: Geometry, algebra, statistics, scientific reasoning

## Key Features

- 6,141 examples from 31 different datasets
- Includes IQTest, FunctionQA, and PaperQA (newly created)
- 9 MathQA and 19 VQA datasets from literature
- Tests logical reasoning on puzzle figures
- Tests algebraic reasoning over functional plots
- Tests scientific reasoning with academic paper figures

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on testmini split
- Supports both multiple-choice and free-form questions
- Answers should be in `\boxed{}` format without units
- Uses numeric equivalence checking for answer comparison
- Chain-of-Thought (CoT) prompting for multiple-choice questions


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `math_vista` |
| **Dataset ID** | [evalscope/MathVista](https://modelscope.cn/datasets/evalscope/MathVista/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `testmini` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 261.48 chars |
| Prompt Length (Min/Max) | 106 / 1391 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,000 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 187x18 - 5236x3491 |
| Formats | jpeg, mpo, png, webp |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "93ec4f16",
      "content": [
        {
          "text": "When a spring does work on an object, we cannot find the work by simply multiplying the spring force by the object's displacement. The reason is that there is no one value for the force-it changes. However, we can split the displacement up in ... [TRUNCATED] ... g of spring constant $k=750 \\mathrm{~N} / \\mathrm{m}$. When the canister is momentarily stopped by the spring, by what distance $d$ is the spring compressed?\nPlease reason step by step, and put your final answer within \\boxed{} without units."
        },
        {
          "image": "[BASE64_IMAGE: jpg, ~185.7KB]"
        }
      ]
    }
  ],
  "target": "1.2",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "precision": 1.0,
    "question_type": "free_form",
    "answer_type": "float",
    "category": "math-targeted-vqa",
    "context": "scientific figure",
    "grade": "college",
    "img_height": 720,
    "img_width": 1514,
    "language": "english",
    "skills": [
      "scientific reasoning"
    ],
    "source": "SciBench",
    "split": "testmini",
    "task": "textbook question answering"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}} without units.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets math_vista \
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
    datasets=['math_vista'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


