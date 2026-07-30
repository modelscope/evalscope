# MiniWoB


## Overview

MiniWoB evaluates whether a multimodal agent can complete short browser tasks such as clicking buttons, filling forms,
scrolling and dragging items.

## Task Description

- **Task Type**: Interactive browser tasks
- **Input**: A task goal, an accessibility tree and a screenshot
- **Output**: Browser actions selected through function calling
- **Dataset**: 125 MiniWoB tasks
- **Metrics**: `success_rate` for completed tasks and `error_rate` for environment failures

## Evaluation Notes

- The default run evaluates one deterministic episode per task.
- Set `repeats=5` for the five-episode schedule.
- Each episode allows up to 10 model/tool turns by default.
- The model must support image input and function calling.
- See the [MiniWoB usage guide](../third_party/miniwob.html) for installation and examples.


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
| Prompt Length (Mean) | 77 chars |
| Prompt Length (Min/Max) | 77 / 77 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "4b7219db",
      "content": "The task goal and browser observation are supplied when the episode is reset."
    }
  ],
  "target": "1",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "browser_action",
      "description": "Execute exactly one BrowserGym MiniWoB action. Supported signatures: noop(wait_ms=1000), mouse_move(x, y), mouse_click(x, y, button=\"left\"), mouse_dblclick(x, y, button=\"left\"), mouse_down(x, y, button=\"left\"), mouse_up(x, y, button=\"left\"),  ... [TRUNCATED 44 chars] ... \"left\"), keyboard_press(key), keyboard_type(text), fill(bid, value). click accepts a string BID, for example click(\"13\"); use mouse_click(x, y) for visual targets. Coordinates are absolute screenshot pixels, not normalized 0-1000 coordinates.",
      "parameters": {
        "properties": {
          "action": {
            "type": "string",
            "description": "One BrowserGym function-call expression."
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
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":10}' \
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
        strategy='function_calling',
        max_steps=10,
    ),
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
