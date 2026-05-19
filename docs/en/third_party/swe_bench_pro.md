# SWE-bench_Pro

## Introduction

[SWE-bench_Pro](https://github.com/scaleapi/SWE-bench_Pro-os) is a more challenging benchmark built by Scale AI on top of SWE-bench, designed to evaluate LLMs / agents on **long-horizon, multilingual, real-world** software engineering tasks. Given a codebase and an issue description, the model drives a multi-turn agent loop inside a per-instance Docker container, autonomously exploring the repository, editing source files and submitting a patch that must pass the hidden unit tests.

```{tip}
**SWE-bench_Pro is strongly recommended over the original SWE-bench.** Compared to `swe_bench_verified` / `swe_bench_lite`:

- **Harder, less contaminated**: samples are drawn from newer, commercial, or non-mainstream repos rather than popular Python projects (Django/Flask/...) that are likely to leak into training data.
- **Multilingual**: covers Python, JavaScript/TypeScript, Go, and more (`repo_language` field), giving a fuller picture of real engineering ability.
- **Longer task horizon**: each instance touches more files and longer modification paths on average, better separating frontier models.
- **Officially-maintained Docker images**: pre-built per-instance images are pulled directly from DockerHub (`jefzda/sweap-images:{tag}`) — **no local image build required**, saving the multi-hour build that the original SWE-bench would require.

The original `swe_bench_*` series is best reserved for historical comparisons or quick smoke tests.
```

## Task Description

- **Task type**: Automated software engineering / bug fixing (agentic)
- **Input**: GitHub issue description (`problem_statement`)
- **Output**: Code patch in unified-diff format, collected after autonomous editing
- **Languages**: Multiple (`repo_language` field; e.g. JavaScript/TypeScript, Python, Go)

## Key Features

- Multi-turn agent loop with a per-instance DockerHub image (`jefzda/sweap-images:{tag}`)
- Sentinel-based patch submission protocol (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`)
- Container-side evaluation: `git apply` patch → run the instance's `run_script.sh` → parse with `parser.py` → check `(fail_to_pass | pass_to_pass) ⊆ PASSED`
- Supports both `toolcall` (function-calling) and `backticks` (text-based fallback for models without function-calling support) action protocols
- **Inference and evaluation share a single sandbox configuration**: `memory_limit` / `cpu_limit` / `platform` are configured once

## Install Dependencies

SWE-bench_Pro uses Docker to ensure reproducibility.

1. **Install Docker**: see the [Docker Installation Guide](https://docs.docker.com/engine/install/).
2. **Linux users**: follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) so you don't need `sudo` every time.
3. Install Python dependencies:

```bash
pip install 'evalscope[sandbox]'
```

`evalscope[sandbox]` installs [`ms-enclave`](https://pypi.org/project/ms-enclave/) and the docker SDK.

```{note}
On the first run, EvalScope will pull each instance's image from DockerHub on demand (~1–4 GB per image) and auto-clone `scaleapi/SWE-bench_Pro-os` into `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` (pinned to commit `ca10a60`) for per-instance `run_script.sh`, `parser.py`, and Dockerfiles.

Make sure you have enough disk space and a stable network.
```

## Run Example

The example below mirrors `test_swe_bench_pro` in [tests/benchmark/test_agent.py](https://github.com/modelscope/evalscope/blob/main/tests/benchmark/test_agent.py):

```python
import os
from evalscope import TaskConfig, run_task
from evalscope.config import SandboxTaskConfig

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['swe_bench_pro'],
    dataset_args={
        'swe_bench_pro': {
            'extra_params': {
                'action_protocol': 'toolcall',  # 'toolcall' or 'backticks'
                'max_steps': 250,               # max agent loop steps
                'command_timeout': 60.0,        # per-bash-command timeout (seconds)
                'eval_timeout': 1800,           # per-instance eval timeout (seconds)
            }
        }
    },
    # Inference and evaluation share this single sandbox config
    sandbox=SandboxTaskConfig(
        default_config={
            'platform': 'linux/amd64',  # default; sweap-images are amd64-only
            'memory_limit': '12g',      # container memory cap; avoids OOM-Killed (e.g. NodeBB)
            'cpu_limit': 4.0,           # container CPU quota (number of CPUs)
        },
    ),
    eval_batch_size=4,  # number of concurrent containers
    limit=5,            # quick smoke test; remove for formal evaluation
    generation_config={
        'temperature': 0.7,
        'parallel_tool_calls': True,
        'stream': True,
    }
)
run_task(task_cfg=task_cfg)
```

Intermediate artifacts are saved under `outputs/<timestamp>/swe_bench_pro_log/<instance_id>/`:
- `workspace/patch.diff`: the patch the model submitted
- `workspace/stdout.log` / `workspace/stderr.log`: test-run logs
- `workspace/output.json`: structured test results parsed by `parser.py`
- `container.log`: container's full execution log

## Parameters

### `extra_params` (dataset-level)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_protocol` | str | `toolcall` | Bash interaction protocol: `toolcall` or `backticks` |
| `max_steps` | int | `250` | Max agent loop steps per sample |
| `command_timeout` | float | `60.0` | Per-bash-command timeout (seconds) |
| `eval_timeout` | int | `3600` | Per-instance evaluation timeout (seconds) |
| `swe_bench_pro_repo_path` | str | `''` | Local path to an existing `SWE-bench_Pro-os` clone; empty = auto-clone and pin to `ca10a60` |
| `dockerhub_username` | str | `jefzda` | DockerHub user/org hosting the sweap-images repository |

### `sandbox.default_config` (task-level, shared by inference and evaluation)

`TaskConfig.sandbox.default_config` is passed straight to ms-enclave's `DockerSandboxConfig` and translated into docker's `mem_limit` / `cpu_quota` etc. when the container is created. Common fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `platform` | str | `linux/amd64` | Container platform. sweap-images only ship for amd64; Apple Silicon must keep this default (runs under QEMU). |
| `memory_limit` | str | `4g` (ms-enclave default) | Container memory cap, e.g. `'8g'`, `'12g'`. **Strongly recommended to set explicitly** — some instances (e.g. NodeBB) easily OOM-Kill their test runs. |
| `cpu_limit` | float | `2.0` (ms-enclave default) | Container CPU quota (number of CPUs). On multi-core hosts, raising to 4+ speeds up tests. |
| `network_enabled` | bool | `true` | Whether the container has network access. |

```{note}
**Why configure memory/CPU under `sandbox.default_config` rather than `extra_params`?**

For SWE-bench_Pro, the inference stage (agent loop) and evaluation stage (running `run_script.sh`) share the **same** sandbox config. Setting it once under `TaskConfig.sandbox.default_config` applies to both, with no duplication.
```

## Troubleshooting

### 1. `OOM-Killed` / container killed mid-test

Some instances (e.g. `nodebb__nodebb-*`) have memory-hungry test suites. Raise `memory_limit`:

```python
sandbox=SandboxTaskConfig(default_config={'memory_limit': '16g', 'cpu_limit': 8.0})
```

### 2. Very slow on Apple Silicon

sweap-images are amd64-only; on Apple Silicon they run under QEMU emulation, **so a single-instance evaluation may take 10–30 minutes**. This is expected. Mitigations:
- Run on an amd64 Linux box; or
- Raise `eval_timeout` (the 3600 s default may be insufficient);
- Lower `eval_batch_size` to avoid host overload from too many concurrent containers.

### 3. `git clone https://github.com/scaleapi/SWE-bench_Pro-os.git` fails

When the network is restricted, the auto-clone will fail. Clone manually and pass the path:

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git /path/to/SWE-bench_Pro-os
git -C /path/to/SWE-bench_Pro-os checkout ca10a60
```

```python
'extra_params': {'swe_bench_pro_repo_path': '/path/to/SWE-bench_Pro-os', ...}
```

### 4. DockerHub rate limiting / pull failures

Anonymous pulls are limited to 100 per 6h. Recommended:

```bash
docker login
```

Or pre-warm common images with `docker pull jefzda/sweap-images:<tag>`.

### 5. `docker_image missing for instance ...`

Usually means `_post_process_samples` did not run, or the sample is missing `repo` / `instance_id`. Confirm the dataset is `ScaleAI/SWE-bench_Pro` and not overridden by a custom dataset args.

### 6. `Missing run_script for instance_xxx`

The `SWE-bench_Pro-os` clone is not at commit `ca10a60`, or some subdirectory is missing. Delete `~/.cache/evalscope/swe_bench_pro/SWE-bench_Pro-os` and let it re-clone, or pass `swe_bench_pro_repo_path` explicitly.

### 7. `pip install evalscope[sandbox]` doesn't seem to install the docker SDK

Verify both docker SDK and ms-enclave are present:

```bash
python -c "import docker, ms_enclave; print(docker.__version__, ms_enclave.__version__)"
```

## Citation

- Paper: [SWE-bench_Pro](https://scale.com/research/swe-bench-pro)
- Repo: <https://github.com/scaleapi/SWE-bench_Pro-os>
- Dataset: <https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro>
