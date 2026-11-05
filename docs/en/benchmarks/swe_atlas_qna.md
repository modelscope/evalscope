# SWE-Atlas-QnA

## Overview

SWE-Atlas QnA is a codebase question-answering benchmark from Scale AI. The public set contains 124 read-only
code-comprehension tasks over 11 production repositories in Go, Python, C, and TypeScript.

## Task Description

- **Task Type**: Multi-turn codebase exploration and question answering
- **Input**: A natural-language question and a task-specific repository image
- **Output**: A prose answer supported by evidence from the repository
- **Domain**: Software engineering and code comprehension

## Key Features

- Runs EvalScope's native AgentLoop with a bash tool inside each task's Docker image.
- Resets the repository to its pinned base commit before evaluation and rejects answers accompanied by source edits.
- Supports positive and negative expert-written rubric items.
- Reports task resolve rate and mean rubric pass rate.

## Evaluation Notes

- `resolve_rate` is the primary metric; a task is resolved when its rubric pass rate reaches `pass_threshold`, which
  defaults to 1.0, and the repository remains clean.
- Each rubric item is graded independently by an LLM judge using the official output schema. Invalid judge replies,
  repository setup failures, and source-tree check failures exclude the sample rather than assigning a zero score.
- A real run requires Docker, `ms-enclave`, and the task images referenced by the dataset. Images can require
  approximately 20-35 GB of disk space in total.
- This adapter uses EvalScope's AgentLoop rather than the official SWE-agent and Modal scaffold, so results are not
  claimed to be leaderboard-equivalent without a separate comparison.

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `swe_atlas_qna` |
| **Dataset ID** | [ScaleAI/SWE-Atlas-QnA](https://modelscope.cn/datasets/ScaleAI/SWE-Atlas-QnA/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `Coding`, `MultiTurn` |
| **Metrics** | `resolve_rate`, `rubric_pass_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 124 |
| Prompt Length (Mean) | 1367.56 chars |
| Prompt Length (Min/Max) | 745 / 3137 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "75d7e494",
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
        "negated": false
      },
      {
        "id": "298d7e5d6d67565d4e30882f238e337c",
        "title": "1.2: Reports that socket paths include the kitty process ID (e.g., /tmp/kitty-12345 for PID 12345).",
        "negated": false
      },
      {
        "id": "30679eccb21e3251eff3613326ff77f4",
        "title": "1.3: Shows captured protocol bytes demonstrating the DCS escape sequence format: \\\\x1bP@kitty-cmd{JSON}\\\\x1b\\\\\\\\ where \\\\x1bP is the DCS introducer, @kitty-cmd is the marker, and \\\\x1b\\\\\\\\ is the String Terminator.",
        "negated": false
      },
      {
        "id": "7e48e77ebe24788711fcdb7b38d5105b",
        "title": "1.4: States that for remote commands, kitty produces no log output - only error conditions trigger log_error() calls.\n",
        "negated": false
      },
      {
        "id": "42b89782d679568bcbe34f9415716cab",
        "title": "1.5: States that command processing flows through these key functions: boss._handle_remote_command() as entry point, parse_cmd() for JSON extraction, handle_cmd() for dispatch, command_for_name() for handler lookup, and the specific command's response_from_kitty() method.",
        "negated": false
      },
      {
        "id": "1782ba2037d6b244f646e31af2e9e12a",
        "title": "1.6: Shows the JSON response structure for the kitten @ ls command with fields (e.g., id, pid, cwd, cmdline, tabs, windows, foreground_processes).\n",
        "negated": false
      },
      {
        "id": "4ca20ef4bbf52c6414a7478d688d08c3",
        "title": "1.7: States that shell integration works through environment variables KITTY_PID, KITTY_LISTEN_ON, and KITTY_WINDOW_ID that kitty sets for child processes.",
        "negated": false
      }
    ]
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
| `working_dir` | `str` | `/app` | Repository path inside each task image. |
| `reset_to_base_commit` | `bool` | `True` | Reset the repository to repository_base_commit before the agent starts. |
| `fail_on_source_modification` | `bool` | `True` | Fail tasks whose repository contains uncommitted changes after the agent run. |
| `command_timeout` | `float` | `900.0` | Timeout in seconds for each sandbox command when no native timeout is configured. |
| `pass_threshold` | `float` | `1.0` | Minimum rubric pass rate required to resolve a task. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets swe_atlas_qna \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":250}' \
    --judge '{"strategy":"llm","models":[{"model_id":"YOUR_JUDGE_MODEL"}]}' \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

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
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
