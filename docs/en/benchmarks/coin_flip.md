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
- F1 score is the primary aggregation metric
- Supports few-shot evaluation with reasoning examples

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `coin_flip` |
| **Dataset ID** | [extraordinarylab/coin-flip](https://modelscope.cn/datasets/extraordinarylab/coin-flip/summary) |
| **Paper** | N/A |
| **Tags** | `Reasoning`, `Yes/No` |
| **Metrics** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |
| **Aggregation** | `f1` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,333 |
| Prompt Length (Mean) | 500.15 chars |
| Prompt Length (Min/Max) | 453 / 551 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "05503706",
      "content": [
        {
          "text": "\nSolve the following coin flip problem step by step. The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem.\n\nQ: A coin is heads up. rushawn flips the coin. yerania ... [TRUNCATED] ...  the coin. jostin does not flip the coin.  Is the coin still heads up?\n\nRemember to put your answer on its own line at the end in the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer YES or NO to the problem.\n\nReasoning:\n"
        }
      ]
    }
  ],
  "target": "NO",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "NO"
  }
}
```

*Note: Some content was truncated for display.*

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


