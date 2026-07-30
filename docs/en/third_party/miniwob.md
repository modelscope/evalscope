# MiniWoB

MiniWoB evaluates multimodal browser agents on short tasks such as clicking, form filling, scrolling and
drag-and-drop. EvalScope runs each episode through an OpenEnv service backed by BrowserGym.

## Installation

```bash
pip install 'evalscope[miniwob]'
```

Docker is required. The first run builds a pinned local image; later runs reuse it. Set
`EVALSCOPE_PIP_INDEX_URL` before evaluation if the image build must use a custom Python package index.

## Quick start

```bash
evalscope eval \
  --model qwen3-vl-plus \
  --datasets miniwob \
  --limit 10 \
  --eval-batch-size 4
```

The default observation contains an accessibility tree and a screenshot. Use a model that accepts images and
supports function calling.

The default dataset has one deterministic episode for each of the 125 tasks. Use `--repeats 5` for the full
five-seed schedule:

```bash
evalscope eval \
  --model qwen3-vl-plus \
  --datasets miniwob \
  --repeats 5 \
  --eval-batch-size 4
```

`limit` is applied before repetition. For example, `--limit 10 --repeats 5` evaluates 50 episodes from 10 tasks.

## Configuration

Most users only need the top-level evaluation parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `repeats` | `1` | Deterministic episodes per task |
| `eval_batch_size` | `1` | Concurrent episodes; do not exceed `4` on typical local machines |
| `limit` | unset | Number of tasks selected before repetition |

Advanced environment options live under `agent_config.task_environment`. `max_steps` remains a native agent option:

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

run_task(
    TaskConfig(
        model='qwen3-vl-plus',
        datasets=['miniwob'],
        repeats=5,
        eval_batch_size=4,
        agent_config=NativeAgentConfig(
            max_steps=10,
            task_environment={
                'backend': 'openenv',
                'observation_mode': 'axtree_screenshot',
                'runtime': {
                    'name': 'ms_enclave_docker',
                },
            },
        ),
    )
)
```

Use `observation_mode='axtree'` only for text-only diagnostics. It does not represent the default multimodal
evaluation.

## Integration structure

`OpenEnvAdapter` owns the reusable episode flow:

1. Start the configured service runtime and obtain its endpoint.
2. Create an OpenEnv session and reset the episode.
3. Run the standard EvalScope AgentLoop.
4. Forward tool actions to `session.step(...)`.
5. Record reward, trace and errors, then close the session and runtime handle.

A benchmark subclass supplies only its dataset schedule, image/environment variables, reset arguments, action mapping
and observation formatting. Action mapping cannot be universal: MiniWoB v0.4.1 accepts a BrowserGym expression in
`action_str`, while other OpenEnv environments such as OpenApp use structured action fields.

The model sees one `browser_action` function. Its `action` argument contains one BrowserGym `miniwob_all` expression,
such as `click("13")`, `fill("7", "text")` or `mouse_click(420, 260)`. Coordinate actions use absolute screenshot
pixels.

## Reproducibility

This integration pins OpenEnv v0.4.1 and BrowserGym v0.14.3. The local image applies a checksum-pinned compatibility
patch so the service uses BrowserGym's `miniwob_all` action configuration and preserves task viewport and timeout
settings.

BrowserGym's full evaluation schedule uses five deterministic seeds per task and a 10-step budget. A limited run,
the default one-seed schedule, or a custom step budget should not be compared directly with full-schedule results.
