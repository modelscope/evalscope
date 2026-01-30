# SWE-bench_Verified


## Overview

SWE-bench Verified is a human-validated subset of 500 samples from SWE-bench, designed to test systems' ability to automatically resolve real-world GitHub issues. Each sample represents a genuine bug fix or feature implementation from popular Python repositories.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing
- **Input**: GitHub issue description with repository context
- **Output**: Code patch (diff format) that resolves the issue
- **Repositories**: 12 popular Python projects (Django, Flask, Requests, etc.)

## Key Features

- 500 human-validated Issue-Pull Request pairs
- Real-world bugs from production Python repositories
- Evaluation via unit test verification
- Docker-based isolated execution environments
- Tests both bug understanding and code modification skills

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically for each repository
- Timeout of 1800 seconds (30 min) per instance
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Supports both local image building and remote image pulling


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_verified` |
| **Dataset ID** | [princeton-nlp/SWE-bench_Verified](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Verified/summary) |
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
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | Oracle dataset ID used to fetch inference context. |
| `build_docker_images` | `bool` | `True` | Build Docker images locally for each sample. |
| `pull_remote_images_if_available` | `bool` | `True` | Attempt to pull existing remote Docker images before building. |
| `force_arch` | `str` | `` | Optionally force the docker images to be pulled/built for a specific architecture. Choices: ['', 'arm64', 'x86_64'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified \
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
    datasets=['swe_bench_verified'],
    dataset_args={
        'swe_bench_verified': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


