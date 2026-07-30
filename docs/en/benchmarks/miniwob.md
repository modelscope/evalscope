# MiniWoB


MiniWoB evaluates multimodal browser agents on 125 short interactive tasks through OpenEnv and BrowserGym.
The default run uses one deterministic seed per task; set `repeats=5` (or `--repeats 5`) for the full five-seed
schedule. Each episode has a default budget of 10 model/tool turns.

The primary metric is `success_rate`; `error_rate` reports environment failures separately. The default observation
contains both an accessibility tree and a screenshot, so the model must support image input and function calling.

See the [MiniWoB usage guide](../third_party/miniwob.html) for installation, runtime configuration, protocol details
and full-schedule examples.


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
      "id": "481bffa9",
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
    "openenv_task_name": "ascending-numbers",
    "observation_mode": "axtree_screenshot",
    "openenv_version": "0.4.1",
    "openenv_commit": "65c506ef94bb1f7279cb4359673b3ef81031d01f",
    "openenv_patch_sha256": "465b23aaf7b3b2cadd681495d694a7dad5ca1b36be0cfb5ce5780b94ac354668",
    "browsergym_version": "0.14.3",
    "browsergym_commit": "0a785fbed075224ae81ca9c1fe924f66050696fe",
    "miniwob_commit": "7fd85d71a4b60325c6585396ec4f48377d049838",
    "csv_sha256": "37117db27909a17b1b78035528472922c98c479a54619ac398dc256a7d2fef09",
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
