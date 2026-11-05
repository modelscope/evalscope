# SWE-Atlas-QnA

## 概述

SWE-Atlas QnA 是 Scale AI 提供的代码库问答基准测试。公开集包含 124 个只读代码理解任务，覆盖 11 个 Go、Python、C 和 TypeScript 生产仓库。

## 任务描述

- **任务类型**：多轮代码库探索与问答
- **输入**：自然语言问题以及特定于任务的仓库镜像
- **输出**：由仓库证据支撑的叙述性答案
- **领域**：软件工程与代码理解

## 主要特点

- 在每个任务的 Docker 镜像内使用 EvalScope 原生 AgentLoop 和 bash 工具。
- 在评估前将仓库重置到固定的基础提交，并拒绝伴随源代码修改的答案。
- 支持专家编写的正面和负面评分项。
- 报告任务解决率和平均评分项通过率。

## 评估说明

- `resolve_rate` 是主要指标；当任务的评分项通过率达到 `pass_threshold`（默认为 1.0）且仓库保持干净时，任务视为已解决。
- 每个评分项由 LLM 评审根据官方输出模式独立评分。无效的评审回复、仓库设置失败以及源代码树检查失败会排除该样本，而不是赋予零分。
- 真实运行需要 Docker、`ms-enclave` 以及数据集引用的任务镜像。镜像总共可能需要约 20-35 GB 的磁盘空间。
- 此适配器使用 EvalScope 的 AgentLoop，而不是官方 SWE-agent 和 Modal 脚手架，因此在没有单独比较的情况下，不声称结果与排行榜等效。

## 属性

| 属性 | 数值 |
|----------|-------|
| **基准测试名称** | `swe_atlas_qna` |
| **数据集ID** | [ScaleAI/SWE-Atlas-QnA](https://modelscope.cn/datasets/ScaleAI/SWE-Atlas-QnA/summary) |
| **论文** | 不适用 |
| **标签** | `Agent`, `Coding`, `MultiTurn` |
| **指标** | `resolve_rate`, `rubric_pass_rate` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 数值 |
|--------|-------|
| 总样本数 | 124 |
| 提示词长度（平均） | 1367.56 字符 |
| 提示词长度（最小/最大） | 745 / 3137 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "5496e73f",
      "content": "I'm  trying to wrap my head around kitty's remote control system because I want to eventually add a new command to it. I've been reading through the code but I'm getting lost in how all the pieces connect.\n\nSo here's what's confusing me. When ... [TRUNCATED 1455 chars] ... ow for a new command.\n\nDon't modify any files in the repository. If you need to create temporary scripts for testing, that's fine. but don't change the actual codebase files. And delete all those temporary scripts/files after task completion."
    }
  ],
  "target": "I started by investigating kitty's remote control system to understand how commands flow from `kitten @ ls` to the running kitty instance and back. \nThe system uses two communication mechanisms: Unix sockets for external connections and TTY-b ... [TRUNCATED 10491 chars] ... nd_for_name()` to handlers in `kitty/rc/`\n5. Environment variables `KITTY_LISTEN_ON`, `KITTY_PID`, `KITTY_WINDOW_ID` for shell integration\n\nNo source files were modified during this investigation. All the temporary test files were cleaned up.",
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
    "task_id": "6905333b74f22949d97ba9e3",
    "prompt": "I'm  trying to wrap my head around kitty's remote control system because I want to eventually add a new command to it. I've been reading through the code but I'm getting lost in how all the pieces connect.\n\nSo here's what's confusing me. When ... [TRUNCATED 1455 chars] ... ow for a new command.\n\nDon't modify any files in the repository. If you need to create temporary scripts for testing, that's fine. but don't change the actual codebase files. And delete all those temporary scripts/files after task completion.",
    "reference_answer": "I started by investigating kitty's remote control system to understand how commands flow from `kitten @ ls` to the running kitty instance and back. \nThe system uses two communication mechanisms: Unix sockets for external connections and TTY-b ... [TRUNCATED 10491 chars] ... nd_for_name()` to handlers in `kitty/rc/`\n5. Environment variables `KITTY_LISTEN_ON`, `KITTY_PID`, `KITTY_WINDOW_ID` for shell integration\n\nNo source files were modified during this investigation. All the temporary test files were cleaned up.",
    "repository_url": "kovidgoyal/kitty",
    "repository_base_commit": "815df1e210e0a9ab4622f5c7f2d6891d7dbeddf1",
    "language": "c",
    "category": "Architecture & system design",
    "docker_image": "ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_kovidgoyal_kitty_1.0",
    "rubric": [
      {
        "id": "a33fc01cba19849aaf3b55e6b801001c",
        "title": "1.1: States that kitty's remote control system uses two communication mechanisms: Unix sockets (when --listen-on or KITTY_LISTEN_ON is configured) and TTY-based DCS escape sequences (when running inside a kitty window).",
        "negated": false,
        "importance": "must have"
      },
      {
        "id": "298d7e5d6d67565d4e30882f238e337c",
        "title": "1.2: Reports that socket paths include the kitty process ID (e.g., /tmp/kitty-12345 for PID 12345).",
        "negated": false,
        "importance": "must have"
      },
      {
        "id": "30679eccb21e3251eff3613326ff77f4",
        "title": "1.3: Shows captured protocol bytes demonstrating the DCS escape sequence format: \\\\x1bP@kitty-cmd{JSON}\\\\x1b\\\\\\\\ where \\\\x1bP is the DCS introducer, @kitty-cmd is the marker, and \\\\x1b\\\\\\\\ is the String Terminator.",
        "negated": false,
        "importance": "must have"
      },
      {
        "id": "7e48e77ebe24788711fcdb7b38d5105b",
        "title": "1.4: States that for remote commands, kitty produces no log output - only error conditions trigger log_error() calls.\n",
        "negated": false,
        "importance": "must have"
      },
      {
        "id": "42b89782d679568bcbe34f9415716cab",
        "title": "1.5: States that command processing flows through these key functions: boss._handle_remote_command() as entry point, parse_cmd() for JSON extraction, handle_cmd() for dispatch, command_for_name() for handler lookup, and the specific command's response_from_kitty() method.",
        "negated": false,
        "importance": "must have"
      },
      {
        "id": "1782ba2037d6b244f646e31af2e9e12a",
        "title": "1.6: Shows the JSON response structure for the kitten @ ls command with fields (e.g., id, pid, cwd, cmdline, tabs, windows, foreground_processes).\n",
        "negated": false,
        "importance": "must have"
      },
      {
        "id": "4ca20ef4bbf52c6414a7478d688d08c3",
        "title": "1.7: States that shell integration works through environment variables KITTY_PID, KITTY_LISTEN_ON, and KITTY_WINDOW_ID that kitty sets for child processes.",
        "negated": false,
        "importance": "must have"
      }
    ]
  }
}
```

*注：为显示方便，部分内容已截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `working_dir` | `str` | `/app` | 每个任务镜像内的仓库路径。 |
| `reset_to_base_commit` | `bool` | `True` | 在代理启动前将仓库重置到 repository_base_commit。 |
| `fail_on_source_modification` | `bool` | `True` | 如果代理运行后仓库包含未提交的更改，则使任务失败。 |
| `command_timeout` | `float` | `900.0` | 当未配置原生超时时，每个沙箱命令的超时时间（秒）。 |
| `pass_threshold` | `float` | `1.0` | 解决任务所需的最低评分项通过率。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_atlas_qna \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":250}' \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['swe_atlas_qna'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=250,
    ),
    judge={'strategy': 'llm', 'models': [{'model_id': 'YOUR_JUDGE_MODEL'}]},
    dataset_args={
        'swe_atlas_qna': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
