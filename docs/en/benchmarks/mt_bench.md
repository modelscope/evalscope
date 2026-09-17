# MT-Bench


## Overview

MT-Bench evaluates chat assistants on challenging, open-ended multi-turn conversations. It was introduced with the
LLM-as-a-judge study and uses a strong judge model to measure response quality beyond closed-form benchmarks.

## Task Description

- **Task Type**: Two-turn open-ended conversational generation with LLM-as-a-judge scoring
- **Input**: An initial user request followed by a user follow-up that depends on the first assistant response
- **Output**: Assistant responses to both turns, with the full conversation preserved for second-turn grading
- **Domain**: Writing, roleplay, extraction, reasoning, mathematics, coding, STEM, and humanities

## Key Features

- Contains 80 human-authored conversations, with 10 two-turn questions in each of eight categories.
- The adapter preserves the generated first answer in the second-turn context, matching the official conversation flow.
- The official protocol uses category temperatures and a maximum of 1,024 generated tokens when no task-level maximum
  is configured.
- The ModelScope mirror supplies the reference answers used for the official reference-guided reasoning, math, and
  coding grading prompts. Its one missing coding reference is supplied from the official FastChat evaluator.

## Evaluation Notes

- The official MT-Bench protocol uses `gpt-4` as the single-answer judge and reports the mean of all valid first- and
  second-turn ratings on a 1-10 scale.
- Configure `judge.models` to supply `gpt-4` for official-comparable runs. Compatible custom judge models are allowed,
  but their scores must not be presented as directly comparable to the official GPT-4 results.
- Reasoning, math, and coding prompts include the dataset reference answers; other categories use the general quality
  rubric covering helpfulness, relevance, accuracy, depth, creativity, and detail.
- Judge replies must satisfy EvalScope's JSON output contract. Parse and transport failures exclude the sample rather
  than assigning a zero score.
- Resources: [Paper](https://arxiv.org/abs/2306.05685) |
  [Official implementation](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) |
  [Dataset](https://modelscope.cn/datasets/HuggingFaceH4/mt_bench_prompts)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mt_bench` |
| **Dataset ID** | [HuggingFaceH4/mt_bench_prompts](https://modelscope.cn/datasets/HuggingFaceH4/mt_bench_prompts/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2306.05685) |
| **Tags** | `Coding`, `InstructionFollowing`, `Math`, `MultiTurn`, `QA`, `Reasoning` |
| **Metrics** | `judge_score`, `first_turn_judge_score`, `second_turn_judge_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 80 |
| Prompt Length (Mean) | 299.55 chars |
| Prompt Length (Min/Max) | 38 / 1642 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `writing` | 10 | 211.7 | 126 | 365 |
| `roleplay` | 10 | 306.5 | 140 | 511 |
| `reasoning` | 10 | 279.4 | 77 | 862 |
| `math` | 10 | 156.1 | 38 | 296 |
| `coding` | 10 | 162.1 | 69 | 541 |
| `extraction` | 10 | 959.5 | 385 | 1642 |
| `stem` | 10 | 202.9 | 92 | 319 |
| `humanities` | 10 | 118.2 | 68 | 219 |

## Sample Example

**Subset**: `writing`

```json
{
  "input": [
    {
      "id": "f685cf02",
      "content": "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "writing",
  "metadata": {
    "category": "writing",
    "prompt_id": 44067482,
    "turns": [
      "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
      "Rewrite your previous response. Start every sentence with the letter A."
    ],
    "references": []
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mt_bench \
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
    datasets=['mt_bench'],
    dataset_args={
        'mt_bench': {
            # subset_list: ['writing', 'roleplay', 'reasoning']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
