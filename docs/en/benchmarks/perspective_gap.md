# PerspectiveGap

## Overview

[PerspectiveGap](https://arxiv.org/abs/2606.08878) is a benchmark for multi-agent orchestration prompting and role-specific context routing. It checks whether a model can give each role the fragments it needs while avoiding irrelevant or distracting fragments.

- [Paper](https://arxiv.org/abs/2606.08878)
- [Dataset](https://huggingface.co/datasets/sun1245/PerspectiveGap)
- [Reference implementation and scorer](https://github.com/WhymustIhaveaname/PerspectiveGap)
- [Leaderboard](https://huggingface.co/spaces/sun1245/PerspectiveGap-Leaderboard)

## Tasks

EvalScope registers two zero-shot PerspectiveGap tasks:

| EvalScope task | PerspectiveGap task | Expected model output |
|---|---|---|
| `perspective_gap_role_assignment` | Role assignment | A JSON object mapping each role to needed fragment IDs, for example `{"coder": ["f1"], "reviewer": ["f2"]}` |
| `perspective_gap_prompt_writing` | Prompt writing | Markdown with one H1 section per role. Each section should include only that role's needed fragments, copied verbatim. |

Both tasks report the deterministic `strict_pass` metric from `perspective_gap.scoring`.
The scorer is imported lazily, so listing benchmarks does not require the optional PerspectiveGap package.

## Install the optional scorer

Install the official scorer before running either task:

```bash
pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git'
```

or, when using uv:

```bash
uv pip install 'perspective-gap @ git+https://github.com/WhymustIhaveaname/PerspectiveGap.git'
```

From an EvalScope source checkout that includes this benchmark, you can also install the benchmark extra:

```bash
pip install '.[perspective_gap]'
```

## Prepare data

The released data is hosted on Hugging Face as [`sun1245/PerspectiveGap`](https://huggingface.co/datasets/sun1245/PerspectiveGap) with a single `test` split.
Run EvalScope with `dataset_hub='huggingface'` to load it from the hub.

For offline runs, download or mirror the dataset JSONL and pass it as `local_path` through `dataset_args`.
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
| Dataset ID | [`sun1245/PerspectiveGap`](https://huggingface.co/datasets/sun1245/PerspectiveGap) |
| Dataset hub | Hugging Face |
| Split | `test` |
| Metrics | `strict_pass` |
| Shots | 0-shot |
| Tags | `Agent`, `InstructionFollowing` |
