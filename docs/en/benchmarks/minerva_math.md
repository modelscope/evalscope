# Minerva-Math


## Overview

Minerva-Math is a benchmark designed to evaluate advanced mathematical and quantitative reasoning capabilities of language models. It consists of 272 challenging problems sourced primarily from MIT OpenCourseWare courses, covering university and graduate-level STEM subjects.

## Task Description

- **Task Type**: Advanced STEM Problem Solving
- **Input**: University/graduate-level mathematical or scientific problem
- **Output**: Step-by-step solution with final answer
- **Difficulty**: University to graduate level

## Key Features

- 272 challenging problems from MIT OpenCourseWare
- Covers advanced subjects: solid-state chemistry, astronomy, differential equations, special relativity
- University and graduate-level difficulty
- Tests deep mathematical and scientific reasoning
- Problems require multi-step quantitative reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Uses LLM-as-judge for complex answer evaluation
- Problems may require domain-specific knowledge (physics, chemistry, etc.)
- Designed to test the upper limits of model reasoning capabilities


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `minerva_math` |
| **Dataset ID** | [knoveleng/Minerva-Math](https://modelscope.cn/datasets/knoveleng/Minerva-Math/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 272 |
| Prompt Length (Mean) | 492.27 chars |
| Prompt Length (Min/Max) | 129 / 1069 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "715a0fa1",
      "content": "Each of the two Magellan telescopes has a diameter of $6.5 \\mathrm{~m}$. In one configuration the effective focal length is $72 \\mathrm{~m}$. Find the diameter of the image of a planet (in $\\mathrm{cm}$ ) at this focus if the angular diameter of the planet at the time of the observation is $45^{\\prime \\prime}$.\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "Start with:\n\\[\ns=\\alpha f \\text {, }\n\\]\nwhere $s$ is the diameter of the image, $f$ the focal length, and $\\alpha$ the angular diameter of the planet. For the values given in the problem:\n\\[\ns=\\frac{45}{3600} \\frac{\\pi}{180} 7200=\\boxed{1.6} \\mathrm{~cm}\n\\]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "type": "Introduction to Astronomy (8.282J Spring 2006)",
    "idx": 0
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

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets minerva_math \
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
    datasets=['minerva_math'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


