# VTCBench


## Overview

VTCBench (Vision-Text Compression Benchmark) evaluates VLMs' ability to compress visual text.

## Task Description

- **Task Type**: Visual question answering with dual evaluation modes
- **Input**: Either (VTC) image(s) + problem text, or (Text) text context + problem text
- **Output**: Short free-form answer
- **Domain**: General visual comprehension, text-rich image understanding

## Key Features

- Dual evaluation modes: image-based (VTC) and text-based (Text)
- Mode VTC tests the model's visual understanding by feeding images directly
- Mode Text tests the model's text-based reasoning using the image's textual context
- The Gap highlights the model's ability to leverage visual information versus textual context for question answering

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Long context benchmark requires longer interval, set `retry_interval` higher to avoid timeout
- Use `--dataset-args '{"vtcbench": {"extra_params":{"eval_mode":"text"}}}'` to switch modes, default is 'vtc'
- Metrics:
  - **containsAll**/**ROUGE-1-R** for Retrieval and Reasoning subsets
  - **ROUGE-L-R**/**LLM-Judge** for Memory subset
- If you encounter casting offset overflow issues, set `DATASET_TF_BATCH_SIZE=1`


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `vtcbench` |
| **Dataset ID** | [MLLM-CL/VTCBench](https://modelscope.cn/datasets/MLLM-CL/VTCBench/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `MultiModal`, `QA`, `Reasoning`, `Retrieval` |
| **Metrics** | `Rouge` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `vtc` | Evaluation mode: vtc (images+problem) or text (text+problem). Choices: ['vtc', 'text'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets vtcbench \
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
    datasets=['vtcbench'],
    dataset_args={
        'vtcbench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
