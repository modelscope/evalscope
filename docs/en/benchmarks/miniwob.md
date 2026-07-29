# MiniWoB (OpenEnv profile)


## Overview

MiniWoB evaluates browser agents on short interactive tasks such as clicking, form filling, drag-and-drop, and
navigation. EvalScope owns the episode schedule, model loop, scoring, traces, and reports. A pinned OpenEnv v0.4.1
BrowserGym service owns the environment lifecycle and reset/step/reward protocol.

## Evaluation

- The schedule contains 625 procedural episodes: 125 BrowserGym 0.14.3 tasks and five deterministic seeds per task.
- The task catalog is downloaded once from a pinned BrowserGym GitHub commit, checksum-verified, and cached locally.
  No ModelScope or Hugging Face dataset is used.
- The primary metric is `success_rate`; `error_rate` separately reports OpenEnv runtime failures.
- Every episode uses a fixed 20-step action budget.
- The default `observation_mode` is `axtree_screenshot`: every reset and step supplies both the accessibility tree and
  a PNG screenshot. Use `axtree` only when a text-only diagnostic run is explicitly desired.
- Screenshot mode requires a model that accepts image input and supports function calling. A text-only model may reject
  the request, ignore the image, or act using only the incomplete accessibility tree; such scores are not representative
  of the default multimodal profile.

## Action and runtime profile

The local runtime applies an EvalScope-owned, checksum-pinned patch to OpenEnv v0.4.1 so that BrowserGym uses its
official `miniwob_all` action configuration and preserves each MiniWoB task's native viewport and timeout instead of
overriding them with OpenEnv server defaults. BrowserGym itself is not forked or modified. Reports record the OpenEnv
source commit and patch checksum.

The action configuration matches BrowserGym 0.14.3, but the EvalScope profile uses a 20-step budget instead of
BrowserGym Experiments' official 10-step budget. Reports therefore set `official_browsergym_action_config=true` and
`official_browsergym_evaluation_protocol=false`; scores must not be compared directly with the official leaderboard.

## Requirements

Install with
`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'evalscope[miniwob]'`.
MiniWoB currently supports only the local `ms_enclave_docker` runtime, which requires Docker and builds the patched
image from a pinned OpenEnv GitHub commit on first use.
The image installs Python dependencies from the Tsinghua PyPI mirror. `eval_batch_size=4` is the recommended maximum
concurrency.

The generic EvalScope `remote` environment runtime remains available to other environment backends. MiniWoB does not
accept it until EvalScope has a standard capability/profile handshake that can verify the remote action mapping and
source profile.

Local mode:

```python
TaskConfig(model='qwen3-vl-plus', datasets=['miniwob'], eval_batch_size=4)
```


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `miniwob` |
| **Dataset ID** | [BrowserGym](https://github.com/ServiceNow/BrowserGym) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling`, `MultiModal`, `MultiTurn` |
| **Metrics** | `success_rate`, `error_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 625 |
| Prompt Length (Mean) | 85 chars |
| Prompt Length (Min/Max) | 85 / 85 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "f2bedbc8",
      "content": "The task goal and browser observation are supplied when the OpenEnv episode is reset."
    }
  ],
  "target": "1",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "browser_action",
      "description": "This is the only browser tool. Always call the tool named browser_action; never call click, fill, press, or another BrowserGym action as a tool name. Put exactly one OpenEnv BrowserGym action expression in the action string. Supported signatu ... [TRUNCATED 461 chars] ... st be absolute pixels in the supplied screenshot, not normalized 0-1000 coordinates. The observation states the exact screenshot width and height. Examples: mouse_click(420, 260), fill(\"7\", \"text\"), keyboard_press(\"ENTER\"), or scroll(0, 300).",
      "parameters": {
        "properties": {
          "action": {
            "type": "string",
            "description": "Exactly one BrowserGym function-call expression."
          }
        },
        "required": [
          "action"
        ]
      }
    }
  ],
  "metadata": {
    "task_name": "miniwob.ascending-numbers",
    "miniwob_category": "hidden test",
    "comment": "",
    "webgum_subset": "False",
    "similarity_group": "0",
    "browsergym_split": "test",
    "task_id": "miniwob.ascending-numbers",
    "openenv_task_name": "ascending-numbers",
    "seed": 28,
    "repeat": 0,
    "profile": "openenv_v0.4.1_miniwob_all_20_steps",
    "max_steps": 20,
    "official_browsergym_action_config": true,
    "official_browsergym_evaluation_protocol": false,
    "openenv_version": "0.4.1",
    "openenv_commit": "65c506ef94bb1f7279cb4359673b3ef81031d01f",
    "openenv_patch_sha256": "b90bb3f1b91c60a8d4b7c888cccd78f1834754b696448da039e1bba7addd836a",
    "browsergym_version": "0.14.3",
    "browsergym_commit": "0a785fbed075224ae81ca9c1fe924f66050696fe",
    "miniwob_commit": "7fd85d71a4b60325c6585396ec4f48377d049838",
    "csv_sha256": "37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09",
    "runtime_mode": null,
    "observation_mode": "axtree_screenshot"
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
| `observation_mode` | `str` | `axtree_screenshot` | Browser observation supplied after reset and every action. Choices: ['axtree', 'axtree_screenshot'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets miniwob \
    --agent-config '{"mode":"native","strategy":"miniwob_openenv_function_calling","max_steps":20}' \
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
    datasets=['miniwob'],
    agent_config=NativeAgentConfig(
        strategy='miniwob_openenv_function_calling',
        max_steps=20,
    ),
    dataset_args={
        'miniwob': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
