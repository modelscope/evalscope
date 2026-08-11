# CoinFlip

## Overview

CoinFlip is a symbolic reasoning benchmark that tests LLMs' ability to track binary state changes through sequences of actions. Each problem involves determining a coin's final state (heads/tails) after various flipping operations.

## Task Description

- **Task Type**: Symbolic Reasoning / State Tracking
- **Input**: Description of coin flip operations by different people
- **Output**: Final coin state (YES for heads-up, NO for tails-up)
- **Focus**: Binary state tracking and logical inference

## Key Features

- Tests state tracking through action sequences
- Binary reasoning (flip/no-flip) decisions
- Requires careful attention to operator effects
- Evaluates systematic logical reasoning
- Clear, unambiguous answers

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should follow "ANSWER: YES/NO" format
- Five metrics: accuracy, precision, recall, F1, yes_ratio
- Accuracy is the primary metric; precision, recall, F1, and yes_ratio provide supporting diagnostics
- Supports few-shot evaluation with reasoning examples

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `coin_flip` |
| **Dataset ID** | [extraordinarylab/coin-flip](https://modelscope.cn/datasets/extraordinarylab/coin-flip/summary) |
| **Paper** | N/A |
| **Tags** | `Reasoning`, `Yes/No` |
| **Metrics** | `accuracy`, `precision`, `recall`, `f1`, `yes_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |
| **Aggregation** | `f1` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text

Solve the following coin flip problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer YES or NO to the problem.

Reasoning:

```

<details>
<summary>Few-shot Template</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}


Solve the following coin flip problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer YES or NO to the problem.

Reasoning:

```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets coin_flip \
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
    datasets=['coin_flip'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
