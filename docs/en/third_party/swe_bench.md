# SWE-bench

## Introduction

[SWE-bench](https://www.swebench.com/) is a benchmark suite for evaluating Large Language Models (LLMs) on real-world software engineering tasks. Each sample corresponds to a real Issue–PR pair from a GitHub repository, and the model is required to produce a code patch that passes the hidden unit tests.

EvalScope's SWE-bench integration provides two evaluation modes:

- **Oracle single-turn mode**: The retrieved relevant code context is provided to the model in one shot, and the model directly generates a patch.
- **Agentic multi-turn mode**: Only the issue description is provided. The model drives a multi-turn agent loop inside a per-instance SWE-bench Docker container, exploring the codebase via `bash`, editing source files and finally submitting a patch (compatible with the [mini-swe-agent](https://github.com/princeton-nlp/mini-swe-agent) pipeline).

## Comparison Between the Two Modes

| Dimension | Oracle Single-turn | Agentic Multi-turn |
|-----------|--------------------|--------------------|
| Input | `problem_statement` + pre-retrieved code context | `problem_statement` only |
| Context source | Pre-injected via `inference_dataset_id` (oracle / BM25) | Model autonomously explores `/testbed` via `bash` |
| Model interaction | Single request, patch produced directly | Multi-turn agent loop (default cap 250 steps) |
| Tool use | None | `bash` tool, supports `toolcall` / `backticks` protocols |
| Submission mechanism | Model's diff output is the submission | Model prints sentinel `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, system collects `git diff` |
| Docker stage | Evaluation stage only (running tests) | Both inference and evaluation stages |
| Datasets | `swe_bench_verified` / `swe_bench_verified_mini` / `swe_bench_lite` | `swe_bench_verified_agentic` / `swe_bench_verified_mini_agentic` / `swe_bench_lite_agentic` |
| Cost | Lower, single-turn inference | Higher, multi-turn inference + long-running containers |

**When to choose which**

- **Oracle mode**: Suitable for quickly establishing a baseline, comparing different retrieval strategies, or running low-cost trials when compute/time is limited.
- **Agentic mode**: Suitable for evaluating the model's end-to-end capability as a "software engineering agent" (autonomous code exploration, patch authoring, iterative debugging) — closer to a real development workflow.

## Datasets

### Core evaluation datasets (shared name roots)

These datasets evaluate models through Issue–PR pairing and unit-test verification, ranging from full to curated subsets:

- **SWE-bench Verified (`swe_bench_verified`)**: 500 manually verified samples from the SWE-bench test set, with strictly controlled quality.
- **SWE-bench Verified Mini (`swe_bench_verified_mini`)**: A 50-sample lightweight subset, ~5GB storage (vs ~130GB for the full set), preserving the difficulty distribution of the original.
- **SWE-bench Lite (`swe_bench_lite`)**: 300 Issue–PR pairs from 11 popular Python projects.

### Oracle-only: inference datasets (`inference_dataset_id`)

Differently-retrieval-formatted SWE-bench versions; they differ in the retrieval method and the size limit on code context:

- **`princeton-nlp/SWE-bench_oracle`**: Oracle retrieval (idealized retrieval as upper-bound baseline). **Default value.**
- **`princeton-nlp/SWE-bench_bm25_13K`**: BM25 retrieval, code context capped at 13,000 tokens (cl100k_base).
- **`princeton-nlp/SWE-bench_bm25_27K`**: BM25 retrieval, capped at 27,000 tokens.
- **`princeton-nlp/SWE-bench_bm25_40K`**: BM25 retrieval, capped at 40,000 tokens.

### Agentic-only: evaluation datasets

Append the `_agentic` suffix to a core dataset name to trigger the agentic multi-turn evaluation:

- **`swe_bench_verified_agentic`** — SWE-bench Verified (500 samples)
- **`swe_bench_verified_mini_agentic`** — SWE-bench Verified Mini (50 samples, ~5GB)
- **`swe_bench_lite_agentic`** — SWE-bench Lite (300 samples)

## Install Dependencies

SWE-bench uses Docker to ensure evaluation reproducibility.

1. **Install Docker**: Refer to the [Docker Installation Guide](https://docs.docker.com/engine/install/) to install Docker on your machine.
2. **Additional configuration for Linux users**: It is recommended to follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for a better experience.
3. Install the Python dependencies:

```bash
pip install evalscope
pip install swebench==4.1.0
```

**Tip**: A properly configured Docker is a prerequisite for running SWE-bench evaluations. Make sure the Docker service is running after installation.

```{note}
On the first run of a swe_bench task, the system needs to build/download the required Docker images, which is resource-intensive:

- **Time**: Building images for the full SWE-bench Verified can take several hours.
- **Storage**: ~130GB for the full set; ~5GB for the Mini set.

**Recommendation**: Make sure you have enough disk space and time budget for the first run, and prefer an environment with a stable network.
```

## Oracle Single-turn Evaluation

Below is an example of evaluating `swe_bench_verified` with the `qwen-plus` model:

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # Use API model service
    datasets=['swe_bench_verified'],  # Can also be 'swe_bench_verified_mini' or 'swe_bench_lite'
    dataset_args={
        'swe_bench_verified': {
            'extra_params': {
                'build_docker_images': True,                              # Build Docker images required for evaluation. Recommended for first run.
                'pull_remote_images_if_available': True,                  # Prefer pulling pre-built remote images when available. Recommended.
                'inference_dataset_id': 'princeton-nlp/SWE-bench_oracle'  # Inference dataset: BM25_13K / 27K / 40K / oracle
            }
        }
    },
    eval_batch_size=5,  # Inference batch size / number of parallel evaluation workers (Docker containers)
    limit=5,            # Limit samples for quick testing; remove for formal evaluation
    generation_config={
        'temperature': 0.1,
    }
)
run_task(task_cfg=task_cfg)
```

Intermediate evaluation artifacts are saved under `outputs/xxxxx/swebench_log`, including `patch.diff` and other files. Example final result:

```text
+-----------+--------------------+----------+----------+-------+---------+---------+
| Model     | Dataset            | Metric   | Subset   |   Num |   Score | Cat.0   |
+===========+====================+==========+==========+=======+=========+=========+
| qwen-plus | swe_bench_verified | mean_acc | default  |     5 |     0.2 | default |
+-----------+--------------------+----------+----------+-------+---------+---------+
```

## Agentic Multi-turn Evaluation

In agentic mode, the model receives only the raw `problem_statement` (no oracle context) and drives a multi-turn agent loop inside a per-instance SWE-bench Docker container: it uses `bash` to explore `/testbed`, edit source files, and finally submits a `git diff` by printing the sentinel `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

### Action Protocol

The `action_protocol` parameter controls how the model interacts with bash:

| Protocol | Description | Best for |
|----------|-------------|----------|
| `toolcall` (default) | OpenAI function-calling with a single `bash` tool, supports parallel tool calls | Models with function-calling support |
| `backticks` | Model wraps commands in `` ```mswea_bash_command ... ``` `` fenced blocks, one command per turn | Models without function-calling |

### Run Example

The example below mirrors `test_swe_bench_verified_mini_agentic` in [tests/benchmark/test_agent.py](https://github.com/modelscope/evalscope/blob/main/tests/benchmark/test_agent.py):

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen3-max',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['swe_bench_verified_mini_agentic'],  # Can also be 'swe_bench_verified_agentic' or 'swe_bench_lite_agentic'
    dataset_args={
        'swe_bench_verified_mini_agentic': {
            'extra_params': {
                'action_protocol': 'toolcall',           # Action protocol: 'toolcall' or 'backticks'
                'max_steps': 250,                        # Maximum agent loop steps
                'command_timeout': 60.0,                 # Per-bash-command timeout (seconds)
                'build_docker_images': True,             # Prepare images required for evaluation; recommended for first run
                'pull_remote_images_if_available': True, # Prefer pulling pre-built remote images
                # 'force_arch': 'arm64',                 # Apple Silicon users may explicitly set 'arm64'; default '' = auto-detect
            }
        }
    },
    eval_batch_size=5,  # Number of parallel containers
    limit=3,            # Limit samples for quick testing; remove for formal evaluation
    generation_config={
        'temperature': 0.7,
        'parallel_tool_calls': True,
        'stream': True,
    }
)
run_task(task_cfg=task_cfg)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_protocol` | str | `toolcall` | Bash interaction protocol: `toolcall` or `backticks` |
| `max_steps` | int | `250` | Maximum agent loop steps per sample |
| `command_timeout` | float | `60.0` | Per-bash-command timeout in seconds |
| `build_docker_images` | bool | `True` | Whether to prepare Docker images (build or pull) |
| `pull_remote_images_if_available` | bool | `True` | Prefer pulling pre-built remote images when available |
| `force_arch` | str | `''` | Force the image architecture: `''` / `arm64` / `x86_64`. Empty means auto-detect. |

```{note}
Agentic mode requires Docker to be running during **both** the inference and evaluation stages. Each sample spawns its own container from a pre-built SWE-bench image, so `eval_batch_size` effectively controls the number of concurrently running containers — tune it according to your machine resources.
```
