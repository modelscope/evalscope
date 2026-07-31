# MiniWoB

MiniWoB evaluates multimodal browser agents on short tasks such as clicking, form filling, scrolling and
drag-and-drop. EvalScope runs BrowserGym directly and uses the environment reward as the success signal.

## Installation

```bash
pip install 'evalscope[miniwob]'
playwright install chromium
```

The first command installs the Python dependencies. Chromium is a platform-specific browser managed by Playwright, so
it cannot be included in the Python extra and must be installed once with the second command. On first use, EvalScope
also downloads and caches the MiniWoB task pages. Docker is not required.

## Quick start

```bash
evalscope eval \
  --model qwen3-vl-plus \
  --datasets miniwob \
  --limit 10 \
  --eval-batch-size 4
```

The observation contains an accessibility tree and a screenshot. Use a model that accepts images and supports
function calling.

The default dataset has one deterministic episode for each of the 125 tasks. Use `--repeats 5` for the full five-seed
schedule:

```bash
evalscope eval \
  --model qwen3-vl-plus \
  --datasets miniwob \
  --repeats 5 \
  --eval-batch-size 4
```

`limit` is applied before repetition. For example, `--limit 10 --repeats 5` evaluates 50 episodes from 10 tasks.

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `repeats` | `1` | Deterministic episodes per task |
| `eval_batch_size` | `1` | Concurrent model calls; BrowserGym operations remain serialized |
| `limit` | unset | Number of tasks selected before repetition |
| `agent_config.max_steps` | `10` | Model/tool turns per episode |

The model sees one `browser_action` function. Its `action` argument contains one BrowserGym `miniwob_all` expression,
such as `click("13")`, `fill("7", "text")` or `mouse_click(420, 260)`. Coordinate actions use absolute screenshot
pixels.

## Evaluation protocol

Each episode runs in a fresh browser context. The task is successful when MiniWoB returns a positive reward.
`success_rate` reports completed tasks, while `error_rate` reports episodes that could not run normally.

The full schedule uses five deterministic episodes per task and a 10-step budget. Results from a limited run, the
default one-episode schedule or a custom step budget should not be compared directly with full-schedule results.
