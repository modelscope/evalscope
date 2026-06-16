# SWE-bench_Lite_Agentic


## Overview

SWE-bench Lite Agentic is the agentic-mode evaluation of SWE-bench Lite, a focused subset of SWE-bench containing 300 Issue-Pull Request pairs from 11 popular Python repositories. The model autonomously drives a multi-turn agent loop inside a per-instance Docker container to resolve real-world GitHub issues.

## Task Description

- **Task Type**: Automated Software Engineering / Bug Fixing (Agentic)
- **Input**: GitHub issue description (no oracle file context)
- **Output**: Code patch (diff format) collected from `git diff` after autonomous editing
- **Size**: 300 carefully selected test instances

## Key Features

- 300 test Issue-Pull Request pairs
- 11 popular Python repositories covered
- Real-world bugs with verified solutions
- Multi-turn agent loop with per-instance Docker sandbox
- More manageable than full SWE-bench while still challenging

## Evaluation Notes

- Requires `pip install swebench==4.1.0` before evaluation
- Docker images are built/pulled automatically for each repository
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/swe_bench.html) for detailed setup instructions
- Popular benchmark variant for initial agentic model comparison

## Agentic Mode

This benchmark drives a multi-turn agent loop (mirrors mini-swe-agent's
`swebench.yaml`) inside a per-instance SWE-bench Docker container. The
model issues `bash` commands to explore `/testbed`, edit source files,
and finally submits its `git diff` patch by printing the sentinel
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` followed by the patch contents.

`extra_params.action_protocol` selects between:
- `toolcall` (default): OpenAI function-calling protocol with a single
  `bash` tool. Recommended for any model that supports tool calling.
- `backticks`: text-based fallback expecting one
  ` ```mswea_bash_command ``` ` block per turn. For models without
  function-calling support.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_bench_lite_agentic` |
| **Dataset ID** | [princeton-nlp/SWE-bench_Lite](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Lite/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 300 |
| Prompt Length (Mean) | 1661.18 chars |
| Prompt Length (Min/Max) | 230 / 24770 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "c8f45390",
      "content": "Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels\nConsider the following model:\r\n\r\n```python\r\nfrom astropy.modeling import models as m\r\nfrom astropy.modeling.separable import separability_matri ... [TRUNCATED 762 chars] ...       [ True,  True, False, False],\r\n       [False, False,  True,  True],\r\n       [False, False,  True,  True]])\r\n```\r\nSuddenly the inputs and outputs are no longer separable?\r\n\r\nThis feels like a bug to me, but I might be missing something?\n"
    }
  ],
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "bash",
      "description": "Execute a bash command inside the sandbox environment. Returns the combined stdout / stderr output of the command.",
      "parameters": {
        "properties": {
          "command": {
            "type": "string",
            "description": "The bash command to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "command"
        ]
      }
    }
  ],
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

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_protocol` | `str` | `toolcall` | Agent action protocol: "toolcall" (mainline OpenAI function-calling, mirrors mini-swe-agent swebench.yaml) or "backticks" (textbased mswea_bash_command fallback for models without function-calling support). Choices: ['toolcall', 'backticks'] |
| `max_steps` | `int` | `250` | Maximum number of agent steps per sample. |
| `command_timeout` | `float` | `60.0` | Default per-bash-command timeout in seconds. |
| `build_docker_images` | `bool` | `True` | Build Docker images locally for each sample. |
| `pull_remote_images_if_available` | `bool` | `True` | Attempt to pull existing remote Docker images before building. |
| `force_arch` | `str` | `` | Optionally force a specific architecture for image build/pull. Choices: ['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | DockerHub user/org namespace for remote SWE-bench images. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_lite_agentic \
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
    datasets=['swe_bench_lite_agentic'],
    dataset_args={
        'swe_bench_lite_agentic': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


