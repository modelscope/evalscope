# AlpacaEval2.0


## Overview

AlpacaEval 2.0 is an evaluation framework for instruction-following language models that uses an LLM judge to compare model outputs against a strong baseline. It provides win-rate metrics reflecting human preferences.

## Task Description

- **Task Type**: Instruction-Following Evaluation (Pairwise Comparison)
- **Input**: User instruction/question
- **Output**: Model response compared against GPT-4 Turbo baseline
- **Metric**: Win rate against baseline model

## Key Features

- Auto-annotator for scalable evaluation
- Compares against GPT-4 Turbo baseline outputs
- High correlation with human preferences
- Cost-effective evaluation method
- Tests general instruction-following capabilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM judge (default: gpt-4-1106-preview)
- Baseline model: gpt-4-turbo outputs
- Reports win rate metric
- Note: Length-controlled win rate not currently supported


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `alpaca_eval` |
| **Dataset ID** | [AI-ModelScope/alpaca_eval](https://modelscope.cn/datasets/AI-ModelScope/alpaca_eval/summary) |
| **Paper** | N/A |
| **Tags** | `Arena`, `InstructionFollowing` |
| **Metrics** | `winrate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `eval` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 805 |
| Prompt Length (Mean) | 164.92 chars |
| Prompt Length (Min/Max) | 12 / 1917 chars |

## Sample Example

**Subset**: `alpaca_eval_gpt4_baseline`

```json
{
  "input": [
    {
      "id": "95236545",
      "content": "What are the names of some famous actors that started their careers on Broadway?"
    }
  ],
  "target": "Several famous actors started their careers on Broadway before making it big in film and television. Here are a few notable examples:\n\n1. Sarah Jessica Parker - Before she was Carrie Bradshaw on \"Sex and the City,\" Sarah Jessica Parker was a  ... [TRUNCATED] ... f the many performers who have transitioned from the Broadway stage to broader fame in the entertainment industry. Broadway often serves as a proving ground for talent, and many actors continue to return to the stage throughout their careers.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "generator": "gpt4_1106_preview",
    "dataset": "helpful_base"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets alpaca_eval \
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
    datasets=['alpaca_eval'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


