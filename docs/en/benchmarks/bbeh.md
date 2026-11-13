# BBEH

## Overview

BIG-Bench Extra Hard (BBEH) replaces each of the 23 tasks in BIG-Bench Hard with a new task that probes a similar reasoning skill but is much harder. Frontier models that saturate BBH score far lower on BBEH, so it separates general reasoning ability across models again.

## Task Description

- **Task Type**: General reasoning (free-form and multiple-choice answers)
- **Input**: A task-specific question followed by the official answer-format instruction
- **Output**: Step-by-step reasoning ending in `The answer is: <answer>`
- **Domain**: Logical, temporal, spatial, causal, arithmetic, linguistic and commonsense reasoning, humour and sarcasm understanding

## Key Features

- 4,520 examples over 23 tasks (200 per task, 120 for Disambiguation QA)
- BBEH Mini: the official 460-example subset (20 per task), selected with `extra_params={'mini': True}`
- Answers are extracted and matched with the official `evaluate.py` from google-deepmind/bbeh
- Long inputs: prompts average about 4,300 characters and reach 32,000 (Shuffled Objects and Zebra Puzzles average over 11,000)

## Evaluation Notes

- 0-shot, with the prompt suffix used in the BBEH paper.
- `acc` per task uses the official fuzzy match (case-insensitive, `(a)` vs `a`, numeric equality, LaTeX `\boxed{}` and markup removal).
- The overall `acc` is the micro average over all examples, which the BBEH leaderboard reports as "Micro Avg".
- The report adds the paper's headline metric, `accuracy` with `harmonic_mean` aggregation: the harmonic mean of per-task accuracies, with 1 percentage point added to each to handle zero scores, as in the paper.
- Give models a generous `max_tokens`; the paper attributes below-random scores to answers cut off before the final line.

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bbeh` |
| **Dataset ID** | [evalscope/bbeh](https://modelscope.cn/datasets/evalscope/bbeh/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2502.19187) |
| **Tags** | `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,520 |
| Prompt Length (Mean) | 4285.09 chars |
| Prompt Length (Min/Max) | 663 / 32506 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `boardgame_qa` | 200 | 7193.37 | 2269 | 14301 |
| `boolean_expressions` | 200 | 3120.27 | 2352 | 4165 |
| `buggy_tables` | 200 | 5448.78 | 4033 | 7202 |
| `causal_understanding` | 200 | 1349.26 | 663 | 2619 |
| `disambiguation_qa` | 120 | 1193.55 | 799 | 2202 |
| `dyck_languages` | 200 | 3277.7 | 2167 | 10978 |
| `geometric_shapes` | 200 | 2836.78 | 1836 | 4897 |
| `hyperbaton` | 200 | 4999.72 | 2751 | 7367 |
| `linguini` | 200 | 1344.07 | 726 | 2138 |
| `movie_recommendation` | 200 | 1504.48 | 1144 | 2172 |
| `multistep_arithmetic` | 200 | 1423.19 | 1187 | 1689 |
| `nycc` | 200 | 1297.05 | 1074 | 1514 |
| `object_counting` | 200 | 9205.75 | 5899 | 12884 |
| `object_properties` | 200 | 5714.45 | 4397 | 7127 |
| `sarc_triples` | 200 | 1717.4 | 1524 | 2467 |
| `shuffled_objects` | 200 | 13838.18 | 10574 | 17484 |
| `spatial_reasoning` | 200 | 6764.89 | 1116 | 32506 |
| `sportqa` | 200 | 2169.74 | 1194 | 3493 |
| `temporal_sequence` | 200 | 4156.03 | 2831 | 5497 |
| `time_arithmetic` | 200 | 1569.47 | 1030 | 2402 |
| `web_of_lies` | 200 | 2959.66 | 1491 | 4320 |
| `word_sorting` | 200 | 2436.12 | 733 | 6989 |
| `zebra_puzzles` | 200 | 11800.46 | 5318 | 15625 |

## Sample Example

**Subset**: `boardgame_qa`

```json
{
  "input": [
    {
      "id": "e18d4e3b",
      "content": "A few players are playing a boardgame. The current state of the game is as follows. The bee has a football with a radius of 15 inches. The chihuahua has a banana-strawberry smoothie. The chihuahua lost her keys. The cobra suspects the truthfu ... [TRUNCATED 9092 chars] ... rkup. For instance: \"The answer is: 42\" or \"The answer is: yes\". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, \"The answer is: (a)\"."
    }
  ],
  "target": "disproved",
  "id": 0,
  "group_id": 0,
  "subset_key": "boardgame_qa",
  "metadata": {
    "mini": false
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}

Think step by step, and when you provide the final answer, please use the prefix "The answer is:" without any modification, and provide the answer directly, with no formatting, no bolding, and no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question is multiple choice with a single correct answer, the final answer must only be the letter corresponding to the correct answer. For example, "The answer is: (a)".
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mini` | `bool` | `False` | Evaluate BBEH Mini, the official 460-example subset (20 per task). |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bbeh \
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
    datasets=['bbeh'],
    dataset_args={
        'bbeh': {
            # subset_list: ['boardgame_qa', 'boolean_expressions', 'buggy_tables']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
