# SWE-bench_Lite


## Overview

SWE-bench Lite is a focused subset of SWE-bench containing 300 Issue-Pull Request pairs from 11 popular Python repositories. It provides a more accessible entry point for evaluating automated software engineering capabilities.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing
- **Input**: GitHub issue description with repository context
- **Output**: Code patch (diff format) that resolves the issue
- **Size**: 300 carefully selected test instances

## Key Features

- 300 test Issue-Pull Request pairs
- 11 popular Python repositories covered
- Real-world bugs with verified solutions
- Evaluation via unit test verification
- More manageable than full SWE-bench while still challenging

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically for each repository
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Popular benchmark variant for initial model comparison


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_lite` |
| **Dataset ID** | [princeton-nlp/SWE-bench_Lite](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Lite/summary) |
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
    --datasets swe_bench_lite \
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
    datasets=['swe_bench_lite'],
    dataset_args={
        'swe_bench_lite': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


