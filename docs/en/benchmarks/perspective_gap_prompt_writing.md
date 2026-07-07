# PerspectiveGap Prompt Writing

## Overview

PerspectiveGap evaluates whether a model can compose orchestration prompts for multi-agent systems while routing only the context each sub-agent needs.

## Tasks

- `perspective_gap_role_assignment`: select the visible fragment IDs for each role and return a JSON object.
- `perspective_gap_prompt_writing`: write one markdown prompt section per role while including only the needed fragments.

## Data

The benchmark uses the ModelScope dataset `evalscope/PerspectiveGap`, which contains the released `test` split. You can also pass `dataset_args.<task>.local_path` to a local JSONL mirror with the same fields.

## Scoring

Scores are computed by `perspective_gap.scoring` from the official PerspectiveGap repository. The scorer is imported lazily so EvalScope can list benchmarks without installing the optional dependency.

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `perspective_gap_prompt_writing` |
| **Dataset ID** | [evalscope/PerspectiveGap](https://modelscope.cn/datasets/evalscope/PerspectiveGap/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2606.08878) |
| **Tags** | `Agent`, `InstructionFollowing` |
| **Metrics** | `strict_pass`, `net_match_score`, `required_coverage`, `boundary_precision`, `distractor_leakage` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 220 |
| Prompt Length (Mean) | 11398.63 chars |
| Prompt Length (Min/Max) | 3909 / 19086 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "196640b6",
      "content": "I need you to set up a 2-agent pipeline for bug bounty.\nThe two roles are coder and reviewer: coder is responsible for finding bugs, reviewer is responsible for judging whether what the coder found counts as a bug.\nAfter each bug the coder fi ... [TRUNCATED 3723 chars] ... of fragment content. Brief connective text between fragments (e.g., \"Then: ...\", \"Note: ...\") is fine. Format: one markdown section per role, with the role name as an h1 header (e.g. `# coder`). Output only the headered prompts, no preamble.\n"
    }
  ],
  "target": "{\"coder\": [\"f5\", \"f6\", \"f7\", \"f1\"], \"reviewer\": [\"f5\", \"f3\", \"f4\"]}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task": "prompt_writing",
    "evaluation_id": "pg_000__seed_1",
    "scenario_id": "pg_000",
    "shuffle_seed": 1,
    "roles": [
      "coder",
      "reviewer"
    ],
    "fragments": [
      {
        "id": "f1",
        "text": "After finding a bug, submit it to a **fresh** reviewer for review. Do not review your own report.\n\n**After the review returns:** execute all action steps in the reviewer's output.",
        "is_distractor": false
      },
      {
        "id": "f2",
        "text": "Claude responds well to clear, explicit instructions. Being specific about your desired output can help enhance results. If you want \"above and beyond\" behavior, explicitly request it rather than relying on the model to infer this from vague  ... [TRUNCATED 276 chars] ... w it. If they'd be confused, Claude will be too.\n\n- Be specific about the desired output format and constraints.\n- Provide instructions as sequential steps using numbered lists or bullet points when the order or completeness of steps matters.",
        "is_distractor": true
      },
      {
        "id": "f3",
        "text": "Audit the report you are given.",
        "is_distractor": false
      },
      {
        "id": "f4",
        "text": "Check the following dimensions one by one:\n\n| Dimension | Requirement |\n|-----------|-------------|\n| Scope | Within consensus-related code directories, or outside but directly affecting consensus logic |\n| Code references | Cited code snippe ... [TRUNCATED 1134 chars] ... ying victim code\n- Code references inconsistent with the latest testnet branch\n- PoC is incomplete and requires the reviewer to manually assemble it\n- Attack effect is equivalent to a Byzantine node simply going offline, with no amplification",
        "is_distractor": false
      },
      {
        "id": "f5",
        "text": "You are participating in a blockchain consensus-layer bug bounty contest.\n\nRead the contest rules first: ./contest-rules.md",
        "is_distractor": false
      },
      {
        "id": "f6",
        "text": "Find vulnerability in the code and write it up as a report.",
        "is_distractor": false
      },
      {
        "id": "f7",
        "text": "This project has been running stably for many years — any obvious fatal bug would have killed it long ago. So:\n\n- Don't report problems that are obvious at a glance (e.g. \"some map is never erased\") — if it were that obvious, the project team would have fixed it themselves\n- Truly valuable vulnerabilities hide in non-obvious interactions, races, and boundary conditions",
        "is_distractor": false
      }
    ],
    "distractor_id": "f2",
    "reference_need_sets": {
      "coder": [
        "f5",
        "f6",
        "f7",
        "f1"
      ],
      "reviewer": [
        "f5",
        "f3",
        "f4"
      ]
    }
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
    --datasets perspective_gap_prompt_writing \
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
    datasets=['perspective_gap_prompt_writing'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
