# EQ-Bench


## Overview

EQ-Bench is a benchmark for evaluating language models on emotional intelligence tasks. It assesses the ability to predict likely emotional responses of characters in dialogues by rating the intensity of possible emotional reactions.

## Task Description

- **Task Type**: Emotional Intelligence Assessment
- **Input**: Dialogue scenario with characters
- **Output**: Emotion intensity ratings in specific format
- **Domains**: Emotional understanding, social cognition

## Key Features

- Tests ability to predict emotional responses in conversations
- Requires rating intensity of multiple possible emotions
- Uses official EQ-Bench v2 scoring algorithm
- Scoring includes sigmoid scaling for small differences
- Adjustment constant ensures random answers score 0

## Evaluation Notes

- Default evaluation uses the **validation** split
- Primary metric: **EQ-Bench Score** (0-100 scale, reported as 0-1)
- Uses zero-shot evaluation (no few-shot examples)
- Responses must include emotion ratings in specific JSON-like format
- Official algorithm from [Paper](https://arxiv.org/abs/2312.06281) | [Homepage](https://eqbench.com/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `eq_bench` |
| **Dataset ID** | [evalscope/EQ-Bench](https://modelscope.cn/datasets/evalscope/EQ-Bench/summary) |
| **Paper** | N/A |
| **Tags** | `InstructionFollowing` |
| **Metrics** | `eq_bench_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 171 |
| Prompt Length (Mean) | 1550.02 chars |
| Prompt Length (Min/Max) | 922 / 3737 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "97129bc9",
      "content": "Your task is to predict the likely emotional responses of a character in this dialogue:\n\nRobert: Claudia, you've always been the idealist. But let's be practical for once, shall we?\nClaudia: Practicality, according to you, means bulldozing ev ... [TRUNCATED] ... ary:\n\nRemorseful: <score>\nIndifferent: <score>\nAffectionate: <score>\nAnnoyed: <score>\n\n\n[End of answer]\n\nRemember: zero is a valid score, meaning they are likely not feeling that emotion. You must score at least one emotion > 0.\n\nYour answer:"
    }
  ],
  "target": "{'emotion1': 'Remorseful', 'emotion2': 'Indifferent', 'emotion3': 'Affectionate', 'emotion4': 'Annoyed', 'emotion1_score': 2, 'emotion2_score': 3, 'emotion3_score': 0, 'emotion4_score': 5}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reference_answer": {
      "emotion1": "Remorseful",
      "emotion2": "Indifferent",
      "emotion3": "Affectionate",
      "emotion4": "Annoyed",
      "emotion1_score": 2,
      "emotion2_score": 3,
      "emotion3_score": 0,
      "emotion4_score": 5
    },
    "reference_answer_fullscale": {
      "emotion1": "Remorseful",
      "emotion2": "Indifferent",
      "emotion3": "Affectionate",
      "emotion4": "Annoyed",
      "emotion1_score": 0,
      "emotion2_score": "6",
      "emotion3_score": 0,
      "emotion4_score": "7"
    }
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
    --datasets eq_bench \
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
    datasets=['eq_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


