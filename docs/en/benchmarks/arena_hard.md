# ArenaHard


## Overview

ArenaHard is a challenging benchmark that evaluates language models through competitive pairwise comparison. Models are judged against a GPT-4 baseline on difficult tasks requiring reasoning, understanding, and generation capabilities.

## Task Description

- **Task Type**: Competitive Model Evaluation (Arena-style)
- **Input**: Challenging instruction/question
- **Output**: Model response compared against GPT-4-0314 baseline
- **Scoring**: Elo-based rating from pairwise battles

## Key Features

- 500 challenging user prompts
- Two-game battle system (A vs B and B vs A)
- Elo rating calculation for model ranking
- Tests reasoning, instruction-following, and generation
- High correlation with Chatbot Arena rankings

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM judge (default: gpt-4-1106-preview)
- Baseline model: gpt-4-0314 outputs
- Reports win rate and Elo-based scores
- Note: Style-controlled win rate not currently supported


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `arena_hard` |
| **Dataset ID** | [AI-ModelScope/arena-hard-auto-v0.1](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary) |
| **Paper** | N/A |
| **Tags** | `Arena`, `InstructionFollowing` |
| **Metrics** | `winrate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `elo` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 500 |
| Prompt Length (Mean) | 406.36 chars |
| Prompt Length (Min/Max) | 29 / 9140 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "088243c7",
      "content": "Use ABC notation to write a melody in the style of a folk tune."
    }
  ],
  "target": "X:1\nT:Untitled Folk Tune\nM:4/4\nL:1/8\nK:G\n|:G2A2|B2A2|G2E2|D4|E2F2|G2F2|E2C2|B,4|\nA2B2|c2B2|A2F2|E4|D2E2|F2E2|D2B,2|C4:|",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "capability": "ABC Sequence Puzzles & Groups"
  }
}
```

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
    --datasets arena_hard \
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
    datasets=['arena_hard'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


