# SciCode-Verified


## Overview

SciCode-Verified evaluates scientific code generation using expert-corrected SciCode tasks and tests.

## Task Description

- **Task Type**: Scientific Code Generation
- **Input**: Multi-step scientific problem with subproblems
- **Output**: Python code solving each subproblem
- **Domain**: Physics, Mathematics, Materials Science, Biology, and Chemistry

## Key Features

- Version 2.2 has 64 problems and 287 scored steps. Three reference steps provide context for later steps.
- The default prompt includes prior step descriptions and extracted code.

## Evaluation Notes

- Requires Docker. Each step has up to 1,800 seconds per Python environment and passes if either environment succeeds.
- Reports the fraction of solved steps and the fraction of problems with every scored step solved.
- Report results with and without problem background separately to match the paper.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `scicode_verified` |
| **Dataset ID** | [evalscope/SciCode-Verified](https://modelscope.cn/datasets/evalscope/SciCode-Verified/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2608.04975) |
| **Tags** | `Coding` |
| **Metrics** | `main_problem_pass_rate`, `subproblem_pass_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 64 |
| Prompt Length (Mean) | 1055.95 chars |
| Prompt Length (Min/Max) | 1055 / 1056 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "ccd84b40",
      "content": "\nPROBLEM DESCRIPTION:\nYou will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier su ... [TRUNCATED 570 chars] ... olve it as follows:\n 1. Generate scientific background required for the next step, in a comment\n 2. Implement a function to solve the problem provided, using the provided header and return line\n\nThe response must be formatted as ```python```\n"
    },
    {
      "id": "49e10587",
      "content": "5"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_name": "Lanczos",
    "problem_id": "5",
    "problem_description_main": "Create a function performing Lanczos Iteration. It takes a symmetric matrix A a number of iterations m and outputs a new matrix Q with orthonomal columns.",
    "problem_background_main": "Background:\nThe Lanczos iteration is the Arnoldi iteration specialized to the hermitian case. The Lanczos iteration performs a \nreduction procedure of matrix $A$ to Hessenberg form by an orthogonal similarity transformation. This similarity \n ... [TRUNCATED 457 chars] ... & \\beta_{n-1} & \\alpha_n\n\\end{bmatrix}\n\\end{equation*}\nwhere $\\alpha$ and $\\beta$ can be written as:\n$$\n\\alpha_n = h_{n,n} = q_n^TAq_n \n$$\n$$\n\\beta_n = h_{n+1,n} = q_{n+1}^T A q_n\n$$\nwhere $q_n$ are the column vector of the needed $Q$ matrix.",
    "problem_io": "\"\"\"\nInputs:\nA : Matrix, 2d array of arbitrary size M * M\nb : Vector, 1d array of arbitrary size M * 1\nm : integer, m < M\n\nOutputs:\nQ : Matrix, 2d array of size M*(m+1)\n\"\"\"",
    "required_dependencies": "import numpy as np",
    "sub_steps": [
      {
        "step_number": "5.1",
        "step_description_prompt": "Create a function performing Lanczos Iteration. It takes a symmetric matrix A a number of iterations m and outputs a new matrix Q with orthonomal columns.",
        "step_background": "Background:\nThe Lanczos iteration is the Arnoldi iteration specialized to the hermitian case. The Lanczos iteration performs a \nreduction procedure of matrix $A$ to Hessenberg form by an orthogonal similarity transformation. This similarity \n ... [TRUNCATED 457 chars] ... & \\beta_{n-1} & \\alpha_n\n\\end{bmatrix}\n\\end{equation*}\nwhere $\\alpha$ and $\\beta$ can be written as:\n$$\n\\alpha_n = h_{n,n} = q_n^TAq_n \n$$\n$$\n\\beta_n = h_{n+1,n} = q_{n+1}^T A q_n\n$$\nwhere $q_n$ are the column vector of the needed $Q$ matrix.",
        "ground_truth_code": null,
        "function_header": "def lanczos(A, b, m):\n    '''Inputs:\n    A : Matrix, 2d array of arbitrary size M * M\n    b : Vector, 1d array of arbitrary size M * 1\n    m : integer, m < M\n    Outputs:\n    Q : Matrix, 2d array of size M*(m+1)\n    '''",
        "test_cases": [
          "n = 7\nh = 1.0/n\ndiagonal = [2/h for i in range(n)]\ndiagonal_up = [-1/h for i in range(n-1)]\ndiagonal_down = [-1/h for i in range(n-1)]\nA = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)\nb = np.array([0.1,0.2,0.0,0.1,0.0,0.3,0.1])\nm = 5\nassert np.allclose(lanczos(A,b,m), target)",
          "n = 7\nh = 1.0/n\ndiagonal = [1/h for i in range(n)]\ndiagonal_up = [-0.9/h for i in range(n-1)]\ndiagonal_down = [-0.9/h for i in range(n-1)]\nA = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)\nb = np.array([0.1,10.1,0.0,0.5,0.2,0.3,0.5])\nm = 5\nassert np.allclose(lanczos(A,b,m), target)",
          "n = 7\nh = 1.0/n\ndiagonal = [1/h for i in range(n)]\ndiagonal_up = [-9/h for i in range(n-1)]\ndiagonal_down = [-9/h for i in range(n-1)]\nA = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)\nA[:, 0] = 0\nA[0, :] = 0\nA[0, 0] = 1/h\nb = np.array([0.1,0.1,0.0,10,0.0,0.1,0.1])\nm = 4\nassert np.allclose(lanczos(A,b,m), target)"
        ],
        "return_line": "    return Q"
      }
    ],
    "general_solution": null,
    "general_tests": [
      "n = 7\nh = 1.0/n\ndiagonal = [2/h for i in range(n)]\ndiagonal_up = [-1/h for i in range(n-1)]\ndiagonal_down = [-1/h for i in range(n-1)]\nA = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)\nb = np.array([0.1,0.2,0.0,0.1,0.0,0.3,0.1])\nm = 5\nassert np.allclose(lanczos(A,b,m), target)",
      "n = 7\nh = 1.0/n\ndiagonal = [1/h for i in range(n)]\ndiagonal_up = [-0.9/h for i in range(n-1)]\ndiagonal_down = [-0.9/h for i in range(n-1)]\nA = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)\nb = np.array([0.1,10.1,0.0,0.5,0.2,0.3,0.5])\nm = 5\nassert np.allclose(lanczos(A,b,m), target)",
      "n = 7\nh = 1.0/n\ndiagonal = [1/h for i in range(n)]\ndiagonal_up = [-9/h for i in range(n-1)]\ndiagonal_down = [-9/h for i in range(n-1)]\nA = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)\nA[:, 0] = 0\nA[0, :] = 0\nA[0, 0] = 1/h\nb = np.array([0.1,0.1,0.0,10,0.0,0.1,0.1])\nm = 4\nassert np.allclose(lanczos(A,b,m), target)"
    ],
    "dataset_info": {
      "dataset_id": "evalscope/SciCode-Verified",
      "revision": "2412b3af319498b46f1a880b07affb6ca5c52d66",
      "problems_test_jsonl_md5": "e107a70b4de390c25852348c19567d33",
      "test_data_cleaned_h5_md5": "32c5602f2bf08393f917f03ff0b80d40"
    }
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text

PROBLEM DESCRIPTION:
You will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier subproblems. Each subproblem should be solved by providing a Python function that meets the specifications provided.

For each subproblem, you will be provided with the following
 1. a description of the subproblem
 2. a function header, which you must use in your solution implementation
 3. a return line, which you must use in your solution implementation

You must only use the following dependencies to implement your solution:
{required_dependencies}

You MUST NOT import these dependencies anywhere in the code you generate.

For each subproblem provided you must solve it as follows:
 1. Generate scientific background required for the next step, in a comment
 2. Implement a function to solve the problem provided, using the provided header and return line

The response must be formatted as ```python```

```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provide_background` | `bool` | `False` | Include scientific background information written by scientists for the problem in the model's prompt. |
| `prompt_mode` | `str` | `official` | 'official': single-turn prompts with the official SciCode templates (SciCode-Verified harness); 'multi_turn': one conversation, one turn per subproblem (evalscope scicode). |
| `grading_envs` | `str` | `or` | 'or': pass if the step passes under numpy 1.26.4/scipy 1.13.1 or numpy 2.4.2/scipy 1.17.1 (SciCode-Verified); '2024' or '2025': that environment only. |
| `problem_ids` | `list[str]` | `[]` | Evaluate only these problem ids (e.g. ["5", "12"]); empty: all problems. |

## Sandbox Configuration

This benchmark requires a sandbox environment for code execution.

```json
{
  "image": "scicode-verified-benchmark:latest",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  }
}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets scicode_verified \
    --sandbox '{"enabled": true}' \
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
    datasets=['scicode_verified'],
    sandbox={'enabled': True},
    dataset_args={
        'scicode_verified': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
