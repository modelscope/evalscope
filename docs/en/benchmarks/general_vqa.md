# General-VQA


## Overview

General-VQA is a customizable visual question answering benchmark for evaluating multimodal models. It supports OpenAI-compatible message format with flexible image input (local paths, URLs, or base64).

## Task Description

- **Task Type**: Visual Question Answering
- **Input**: Images + questions in OpenAI chat format
- **Output**: Free-form text answer
- **Flexibility**: Supports custom datasets via TSV/JSONL files

## Key Features

- OpenAI-compatible message format
- Supports multiple image input methods (path, URL, base64)
- Flexible evaluation with BLEU and Rouge metrics
- Custom dataset support via local file loading
- Extensible for various VQA use cases

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Default metrics: **BLEU**, **Rouge** (Rouge-L-R as main score)
- Evaluates on **test** split
- See [User Guide](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/vlm.html) for dataset format


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `general_vqa` |
| **Dataset ID** | `general_vqa` |
| **Paper** | N/A |
| **Tags** | `Custom`, `MultiModal`, `QA` |
| **Metrics** | `BLEU`, `Rouge` |
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
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_vqa \
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
    datasets=['general_vqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


