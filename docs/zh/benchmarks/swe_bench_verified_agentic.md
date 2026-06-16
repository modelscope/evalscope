# SWE-bench_Verified_Agentic


## 概述

SWE-bench Verified Agentic 是对 SWE-bench Verified 的代理模式（agentic-mode）评估。SWE-bench Verified 是从 SWE-bench 中人工验证筛选出的 500 个样本子集。与单轮“神谕”（oracle）变体不同，模型必须在每个实例专属的 Docker 容器内，通过多轮代理循环自主探索代码仓库、执行 shell 命令、编辑源文件，并最终提交补丁。

## 任务描述

- **任务类型**：自动化软件工程 / 缺陷修复（代理模式）
- **输入**：GitHub issue 描述（不提供神谕文件上下文）
- **输出**：模型自主编辑后通过 `git diff` 生成的代码补丁（diff 格式）
- **涉及仓库**：12 个流行的 Python 项目（如 Django、Flask、Requests 等）

## 主要特性

- 包含 500 个人工验证的 Issue-Pull Request 对
- 支持多轮代理循环（兼容 mini-swe-agent 的 `swebench.yaml` 配置）
- 每个实例使用独立的 SWE-bench Docker 容器作为执行沙箱
- 基于哨兵值（sentinel）的补丁提交协议（`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`）
- 同时支持函数调用（`toolcall`）和基于文本（`backticks`）的动作协议

## 评估说明

- 评估前需安装 `pip install swebench==4.1.0`
- 每个仓库的 Docker 镜像会自动构建或拉取
- 每个实例的最终补丁验证超时时间为 1800 秒（30 分钟）
- 详细设置说明请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/swe_bench.html)
- 支持本地构建镜像和远程拉取镜像两种方式

## 代理模式

该基准测试在每个实例专属的 SWE-bench Docker 容器内驱动一个多轮代理循环（与 mini-swe-agent 的 `swebench.yaml` 配置一致）。模型通过发出 `bash` 命令来探索 `/testbed` 目录、编辑源文件，并最终通过打印哨兵值 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 及其后的补丁内容来提交 `git diff` 补丁。

`extra_params.action_protocol` 参数用于选择以下两种协议之一：
- `toolcall`（默认）：采用 OpenAI 函数调用协议，仅提供一个 `bash` 工具。推荐用于支持工具调用的模型。
- `backticks`：基于文本的备用协议，要求每轮输出一个 ` ```mswea_bash_command ``` ` 代码块。适用于不支持函数调用的模型。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `swe_bench_verified_agentic` |
| **数据集ID** | [princeton-nlp/SWE-bench_Verified](https://modelscope.cn/datasets/princeton-nlp/SWE-bench_Verified/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均） | 1699.73 字符 |
| 提示词长度（最小/最大） | 143 / 24770 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "360d65af",
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

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `action_protocol` | `str` | `toolcall` | 代理动作协议："toolcall"（主流 OpenAI 函数调用方式，与 mini-swe-agent 的 swebench.yaml 一致）或 "backticks"（针对不支持函数调用的模型的基于文本的 mswea_bash_command 回退方案）。可选值：['toolcall', 'backticks'] |
| `max_steps` | `int` | `250` | 每个样本的最大代理步数。 |
| `command_timeout` | `float` | `60.0` | 每个 bash 命令的默认超时时间（秒）。 |
| `build_docker_images` | `bool` | `True` | 是否为每个样本在本地构建 Docker 镜像。 |
| `pull_remote_images_if_available` | `bool` | `True` | 在构建前是否尝试拉取已存在的远程 Docker 镜像。 |
| `force_arch` | `str` | `` | 可选地强制指定镜像构建/拉取的目标架构。可选值：['', 'arm64', 'x86_64'] |
| `dockerhub_username` | `str` | `swebench` | 远程 SWE-bench 镜像在 DockerHub 上的用户/组织命名空间。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_bench_verified_agentic \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['swe_bench_verified_agentic'],
    dataset_args={
        'swe_bench_verified_agentic': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```