# GSM8K


## Overview

GSM8K (Grade School Math 8K) is a high-quality dataset of 8.5K linguistically diverse grade school math word problems created by human problem writers. The dataset is specifically designed to evaluate and improve the multi-step mathematical reasoning capabilities of language models.

## Task Description

- **Task Type**: Mathematical Word Problem Solving
- **Input**: Natural language math word problem
- **Output**: Numerical answer derived through step-by-step reasoning
- **Difficulty**: Grade school level (2-8 reasoning steps required)

## Key Features

- Problems require basic arithmetic operations (addition, subtraction, multiplication, division)
- Solutions involve 2 to 8 sequential reasoning steps
- High linguistic diversity in problem formulations
- Human-written problems ensuring natural language quality
- Clear numerical answers for objective evaluation

## Evaluation Notes

- Default configuration uses **4-shot** examples with Chain-of-Thought (CoT) prompting
- Answers should be formatted within `\boxed{}` for proper extraction
- The metric extracts numerical values for accuracy comparison
- Supports both zero-shot and few-shot evaluation modes


## Properties

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

*Statistics not available.*

## Sample Example

*Sample example not available.*

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

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gsm8k \
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
    datasets=['gsm8k'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


