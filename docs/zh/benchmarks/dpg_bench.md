# DPG-Bench


## Overview

DPG-Bench (ELLA) evaluates semantic alignment on 1,065 dense prompts describing multiple
objects, attributes and relations, built from COCO, PartiPrompts, DSG-1k and Object365.

## Task Description

- **Task Type**: Text-to-image generation
- **Input**: Dense prompt describing a multi-object scene
- **Output**: Image scored through atomic yes/no questions decomposed from the prompt
- **Domain**: Dense prompt semantic alignment

## Key Features

- 1,065 dense prompts decomposed into ~14k atomic questions with dependency structure,
  following the DSG scoring pipeline of the official evaluator.
- Loads the ModelScope mirror by default; the official metadata CSV (from the
  diffusionbench-evals packaging of the ELLA evaluator) can be used through a local
  ``dataset_id`` while the Hugging Face mirror is being published.

## Evaluation Notes

- A vision API answers each atomic question. An answer scores 1.0 only when it is exactly
  'yes'; a question whose parent is adjusted to 0 is forced to 0. The overall score is the
  mean of adjusted per-image ratios; the five L1 category scores (Global / Entity /
  Attribute / Relation / Other) aggregate RAW answers corpus-level, as in the official
  evaluator.
- Defaults to qwen3.8-flash on DashScope: set DASHSCOPE_API_KEY; no local VLM weights are
  needed. The official evaluator uses mPLUG, so results from different judges are not
  directly comparable. API failures are excluded rather than scored zero. The judge prompt
  pins the accepted token (lowercase 'yes'/'no', no punctuation), matching the strict
  exact-match rule.
- A prompt-level category without questions emits no report row; use repeats=4 for the
  official repeated-image protocol.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `dpg_bench` |
| **Dataset ID** | [evalscope/DPG-Bench](https://modelscope.cn/datasets/evalscope/DPG-Bench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2403.12915) |
| **Tags** | `TextToImage` |
| **Metrics** | `DPGScore` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --model-task image_generation \
    --eval-type text2image \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets dpg_bench \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='text2image',
    model_task='image_generation',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['dpg_bench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
