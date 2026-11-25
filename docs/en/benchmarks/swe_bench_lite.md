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

- Supports official E2B and Aliyun Agent Sandbox through `TaskConfig.sandbox.engine='e2b'`, avoiding local Docker and local image downloads. See the [sandbox guide](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html#e2b).
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
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 300 |
| Prompt Length (Mean) | 57725.21 chars |
| Prompt Length (Min/Max) | 5169 / 372080 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "dcd8fb65",
      "content": "You will be provided with a partial code base and an issue statement explaining a problem to resolve.\n<issue>\nModeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\r\n\r\n ... [TRUNCATED 18305 chars] ... 2.0\n+        while y != y1:\n+            points.append((x, y))\n+            err -= dx\n+            if err < 0:\n+                x += sx\n+                err += dy\n+            y += sy\n \n+    points.append((x, y))\n     return points\n</patch>\n\n"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_statement": "Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\r\n\r\n```python\r\nfrom astropy.modeling import models as m\r\nfrom astropy.modeling.separable import separability_matri ... [TRUNCATED 762 chars] ...       [ True,  True, False, False],\r\n       [False, False,  True,  True],\r\n       [False, False,  True,  True]])\r\n```\r\nSuddenly the inputs and outputs are no longer separable?\r\n\r\nThis feels like a bug to me, but I might be missing something?\n",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
    "patch": "diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n--- a/astropy/modeling/separable.py\n+++ b/astropy/modeling/separable.py\n@@ -242,7 +242,7 @@ def _cstack(left, right):\n         cright = _coord_matrix(right, 'right', noutp)\n     else:\n         cright = np.zeros((noutp, right.shape[1]))\n-        cright[-right.shape[0]:, -right.shape[1]:] = 1\n+        cright[-right.shape[0]:, -right.shape[1]:] = right\n \n     return np.hstack([cleft, cright])\n \n",
    "PASS_TO_PASS": [
      "astropy/modeling/tests/test_separable.py::test_coord_matrix",
      "astropy/modeling/tests/test_separable.py::test_cdot",
      "astropy/modeling/tests/test_separable.py::test_cstack",
      "astropy/modeling/tests/test_separable.py::test_arith_oper",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model0-result0]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model1-result1]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model2-result2]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model3-result3]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model4-result4]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model5-result5]",
      "... [TRUNCATED 3 more items] ..."
    ],
    "FAIL_TO_PASS": [
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]",
      "astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]"
    ],
    "test_patch": "diff --git a/astropy/modeling/tests/test_separable.py b/astropy/modeling/tests/test_separable.py\n--- a/astropy/modeling/tests/test_separable.py\n+++ b/astropy/modeling/tests/test_separable.py\n@@ -28,6 +28,13 @@\n p1 = models.Polynomial1D(1, nam ... [TRUNCATED 931 chars] ...          [True,  True,  False, False, False],\n+                        [False, False, True,  False, False],\n+                        [False, False, False, True,  False],\n+                        [False, False, False, False, True]]))),\n }\n \n \n",
    "version": "4.3",
    "repo": "astropy/astropy",
    "environment_setup_commit": "298ccb478e6bf092953bca67a3d29dc6c35f6752",
    "hints_text": "",
    "created_at": "2022-03-03T15:14:54Z",
    "docker_image": "swebench/sweb.eval.arm64.astropy_1776_astropy-12907:latest"
  }
}
```

*Note: Some content was truncated for display.*

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
| `inference_dataset_id` | `str` | `princeton-nlp/SWE-bench_oracle` | Oracle dataset ID used to fetch inference context from the selected dataset hub. |
| `inference_dataset_revision` | `str | null` | `None` | Optional revision of the oracle inference dataset. |
| `force_arch` | `str` | `` | Optionally force the docker images to be pulled/built for a specific architecture. Choices: ['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | DockerHub user/org namespace for remote SWE-bench images. |

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
