# SWE-bench_Verified_mini


## Overview

SWE-bench Verified Mini is a compact subset of SWE-bench Verified, containing 50 carefully selected samples that maintain the same distribution of performance, test pass rates, and difficulty as the full dataset while requiring only 5GB of storage instead of 130GB.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing
- **Input**: GitHub issue description with repository context
- **Output**: Code patch (diff format) that resolves the issue
- **Size**: 50 samples (vs 500 in full Verified set)

## Key Features

- Representative 50-sample subset of SWE-bench Verified
- Same difficulty distribution as full dataset
- Dramatically reduced storage requirements (5GB vs 130GB)
- Ideal for quick evaluation and development iteration
- Maintains statistical validity for benchmarking

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup
- Good for rapid prototyping and initial model assessment


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_verified_mini` |
| **Dataset ID** | [evalscope/swe-bench-verified-mini](https://modelscope.cn/datasets/evalscope/swe-bench-verified-mini/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `build_docker_images` | `bool` | `True` | Build Docker images locally for each sample. |
| `pull_remote_images_if_available` | `bool` | `True` | Attempt to pull existing remote Docker images before building. |
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | Oracle dataset ID used to fetch inference context. |
| `force_arch` | `str` | `` | Optionally force the docker images to be pulled/built for a specific architecture. Choices: ['', 'arm64', 'x86_64'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified_mini \
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
    datasets=['swe_bench_verified_mini'],
    dataset_args={
        'swe_bench_verified_mini': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


