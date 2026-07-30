# MiniWoB


## Overview

MiniWoB evaluates browser agents on short interactive tasks such as clicking, form filling, drag-and-drop, and
navigation. EvalScope owns the episode schedule, model loop, scoring, traces, and reports. A pinned OpenEnv v0.4.1
BrowserGym service owns the environment lifecycle and reset/step/reward protocol.

## Evaluation

- The default schedule contains 125 procedural episodes: 125 BrowserGym 0.14.3 tasks and one deterministic seed per
  task. Set `TaskConfig.repeats=5` (or pass `--repeats 5` on the CLI) to run the full BrowserGym schedule of 625
  episodes with five distinct deterministic seeds per task.
- The task catalog is downloaded once from a pinned BrowserGym GitHub commit, checksum-verified, and cached locally.
  No ModelScope or Hugging Face dataset is used.
- The primary metric is `success_rate`; `error_rate` separately reports OpenEnv runtime failures.
- Every episode uses a 10-step action budget by default, matching BrowserGym Experiments. Override it with
  `NativeAgentConfig.max_steps` only for diagnostic or custom runs.
- `agent_config.task_environment.observation_mode` controls the observation representation. Its default is
  `axtree_screenshot`: every reset and step supplies both the accessibility tree and a PNG screenshot. Use `axtree`
  only when a text-only diagnostic run is explicitly desired.
- Screenshot mode requires a model that accepts image input and supports function calling. A text-only model may reject
  the request, ignore the image, or act using only the incomplete accessibility tree; such scores are not representative
  of the default multimodal profile.

## Action and runtime profile

The local runtime applies an EvalScope-owned, checksum-pinned patch to OpenEnv v0.4.1 so that BrowserGym uses its
official `miniwob_all` action configuration and preserves each MiniWoB task's native viewport and timeout instead of
overriding them with OpenEnv server defaults. BrowserGym itself is not forked or modified. Reports record the OpenEnv
source commit and patch checksum.

The default action configuration and 10-step budget match BrowserGym 0.14.3. The full BrowserGym evaluation protocol
also requires `TaskConfig.repeats=5` and an untruncated schedule; reports set
`official_browsergym_evaluation_protocol=true` only when all three conditions match. Scores from the lighter one-seed
default, a limited run, or a custom step budget must not be compared directly with the official leaderboard.

## Requirements

Install with `pip install 'evalscope[miniwob]'`.
MiniWoB currently supports only the local `ms_enclave_docker` runtime, which requires Docker and builds the patched
image from a pinned OpenEnv GitHub commit on first use.
Set `EVALSCOPE_PIP_INDEX_URL` before evaluation to use a custom Python package index while building the image.
`eval_batch_size=4` is the recommended maximum concurrency.

Local mode:

```python
TaskConfig(model='qwen3-vl-plus', datasets=['miniwob'], eval_batch_size=4)
```

Full five-seed schedule:

```python
TaskConfig(model='qwen3-vl-plus', datasets=['miniwob'], repeats=5, eval_batch_size=4)
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
| Total Samples | 125 |
| Prompt Length (Mean) | 85 chars |
| Prompt Length (Min/Max) | 85 / 85 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "6f138905",
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
    "profile": "openenv_v0.4.1_miniwob_all_10_steps",
    "max_steps": 10,
    "repeats": 1,
    "official_browsergym_action_config": true,
    "official_browsergym_evaluation_protocol": false,
    "openenv_version": "0.4.1",
    "openenv_commit": "65c506ef94bb1f7279cb4359673b3ef81031d01f",
    "openenv_patch_sha256": "465b23aaf7b3b2cadd681495d694a7dad5ca1b36be0cfb5ce5780b94ac354668",
    "browsergym_version": "0.14.3",
    "browsergym_commit": "0a785fbed075224ae81ca9c1fe924f66050696fe",
    "miniwob_commit": "7fd85d71a4b60325c6585396ec4f48377d049838",
    "csv_sha256": "37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09",
    "runtime_mode": null,
    "observation_mode": "axtree_screenshot",
    "seed": 1608637542,
    "repeat": 0
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets miniwob \
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
    datasets=['miniwob'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
