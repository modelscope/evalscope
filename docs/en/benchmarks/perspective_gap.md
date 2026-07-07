# PerspectiveGap

## Overview

PerspectiveGap is a benchmark for multi-agent orchestration prompting.
It evaluates whether a model can route context to sub-agent prompts without leaking irrelevant or distracting fragments.

Paper: <https://arxiv.org/abs/2606.08878>

Dataset: <https://huggingface.co/datasets/sun1245/PerspectiveGap>

Official scorer: <https://github.com/WhymustIhaveaname/PerspectiveGap>

## Tasks

EvalScope registers two PerspectiveGap tasks:

| EvalScope task | PerspectiveGap task | Expected model output |
|---|---|---|
| `perspective_gap_role_assignment` | role-fragment assignment | A JSON object mapping each role to fragment IDs, for example `{"coder": ["f1"], "reviewer": ["f2"]}` |
| `perspective_gap_prompt_writing` | free-form prompt writing | One markdown section per role, using the role name as an h1 header and pasting the needed fragments verbatim |

Both tasks use the deterministic `strict_pass` metric from `perspective_gap.scoring`.
The scorer is imported lazily, so benchmark discovery does not require the optional PerspectiveGap dependency.

## Install the optional scorer

```bash
pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git'
```

or, when using uv:

```bash
uv pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git'
```

After this PR is installed with extras, the scorer can also be installed through the `perspective_gap` extra.

## Prepare data

The released data is hosted on Hugging Face as `sun1245/PerspectiveGap` with a single `test` split.
Run EvalScope with `dataset_hub='huggingface'`.

If you need an offline run, download or mirror the dataset JSONL and pass it as `local_path` through `dataset_args`.
The local file must preserve the official fields:

- `evaluation_id`
- `scenario_id`
- `shuffle_seed`
- `roles`
- `fragments`
- `distractor_id`
- `reference_need_sets`
- `role_assignment_prompt`
- `prompt_writing_prompt`

## CLI usage

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key YOUR_API_KEY \
    --dataset-hub huggingface \
    --datasets perspective_gap_role_assignment perspective_gap_prompt_writing
```

Use one task at a time if you want separate runs:

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key YOUR_API_KEY \
    --dataset-hub huggingface \
    --datasets perspective_gap_role_assignment
```

## Python usage

```python
from evalscope import run_task
from evalscope.config import TaskConfig
from evalscope.constants import HubType

cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='YOUR_API_KEY',
    dataset_hub=HubType.HUGGINGFACE,
    datasets=[
        'perspective_gap_role_assignment',
        'perspective_gap_prompt_writing',
    ],
)

run_task(task_cfg=cfg)
```

For local JSONL data:

```python
cfg = TaskConfig(
    model='YOUR_MODEL',
    datasets=['perspective_gap_role_assignment'],
    dataset_args={
        'perspective_gap_role_assignment': {
            'local_path': '/path/to/perspectivegap/evaluations.jsonl',
        },
    },
)
```

## Properties

| Property | Value |
|---|---|
| Benchmark names | `perspective_gap_role_assignment`, `perspective_gap_prompt_writing` |
| Dataset ID | `sun1245/PerspectiveGap` |
| Dataset hub | Hugging Face |
| Split | `test` |
| Metrics | `strict_pass` |
| Shots | 0-shot |
| Tags | `Agent`, `InstructionFollowing` |
