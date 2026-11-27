# SciCode-Verified


## 概述

SciCode-Verified 使用经专家修正的 SciCode 任务和测试来评估科学代码生成能力。

## 任务描述

- **任务类型**：科学代码生成
- **输入**：包含子问题的多步科学问题
- **输出**：解决每个子问题的 Python 代码
- **领域**：物理、数学、材料科学、生物学和化学

## 主要特点

- 2.2 版本包含 64 个问题和 287 个计分步骤。其中 3 个参考步骤为后续步骤提供上下文。
- 默认提示词包含先前步骤的描述和提取的代码。

## 评估说明

- 需要 Docker。每个步骤在每个 Python 环境中的时间上限为 1,800 秒，只要任一环境成功即视为通过。
- 报告已解决步骤的比例，以及所有计分步骤均已解决的问题的比例。
- 为了与论文保持一致，请分别报告包含和不包含问题背景的结果。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `scicode_verified` |
| **数据集ID** | [evalscope/SciCode-Verified](https://modelscope.cn/datasets/evalscope/SciCode-Verified/summary) |
| **论文** | [论文](https://arxiv.org/abs/2608.04975) |
| **标签** | `Coding` |
| **指标** | `main_problem_pass_rate`, `subproblem_pass_rate` |
| **默认 Shots** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 64 |
| 提示词长度（平均） | 1055.95 字符 |
| 提示词长度（最小/最大） | 1055 / 1056 字符 |

## 样例示例

**子集**：`default`

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

*注：为便于展示，部分内容已截断。*

## 提示模板

**提示模板：**
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

## 额外参数

| 参数 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `provide_background` | `bool` | `False` | 在模型提示词中包含科学家为该问题撰写的科学背景信息。 |
| `prompt_mode` | `str` | `official` | “official”：使用官方 SciCode 模板的单轮提示词（SciCode-Verified harness）；“multi_turn”：一次对话，每个子问题一轮（evalscope scicode）。 |
| `grading_envs` | `str` | `or` | “or”：如果步骤在 numpy 1.26.4/scipy 1.13.1 或 numpy 2.4.2/scipy 1.17.1 任一环境中通过，则视为通过（SciCode-Verified）；“2024” 或 “2025”：仅使用该环境。 |
| `problem_ids` | `list[str]` | `[]` | 仅评估这些问题 ID（例如 ["5", "12"]）；为空时评估所有问题。 |

## 沙箱配置

该基准测试需要一个沙箱环境来执行代码。

```json
{
  "image": "scicode-verified-benchmark:latest",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  }
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets scicode_verified \
    --sandbox '{"enabled": true}' \
    --limit 10  # Remove this line for formal evaluation
```

### 使用 Python

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
