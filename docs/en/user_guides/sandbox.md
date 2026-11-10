# Sandbox Environments

Sandboxes run model-generated code or provide agents with an isolated command execution environment. EvalScope uses [ms-enclave](https://github.com/modelscope/ms-enclave) to create and manage these environments, releasing resources when evaluation finishes.

## Choose an Execution Environment

| Option | Use case | Prerequisites |
| --- | --- | --- |
| Local Docker | Run code evaluations or agents on your machine | Install and start Docker locally |
| Self-hosted remote Docker | Run execution tasks on your own server | Install Docker and start the ms-enclave service on the server |
| E2B remote sandboxes | Use official E2B or Aliyun Agent Sandbox without maintaining a Docker service | A provider account and an E2B-compatible API key |
| VolcEngine SandboxFusion | Connect to an existing SandboxFusion service | An accessible service endpoint |

Code evaluation benchmarks such as [HumanEval](../benchmarks/humaneval.md) typically provide a default image. For [agent evaluations](agent/index.md), check the benchmark documentation for environment requirements.

## Quick Start: Local Docker

Install [Docker](https://www.docker.com/get-started) and start the service, then install the dependencies:

```bash
pip install 'evalscope[sandbox]'
export DASHSCOPE_API_KEY='<your-model-api-key>'
```

The following example evaluates five HumanEval samples using Docker:

```python
import os

from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    model='qwen3.8-flash',
    eval_type='openai_api',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    datasets=['humaneval'],
    limit=5,
    eval_batch_size=4,
    generation_config={'max_tokens': 4096, 'temperature': 0},
    sandbox={
        'enabled': True,
        'engine': 'docker',
    },
)

run_task(task_config)
```

The first run downloads the image. Adjust concurrency to fit your machine's CPU and memory; multilingual images typically require more resources.

## Shared Configuration

Configure sandboxes through `TaskConfig.sandbox`:

| Field | Purpose | Default |
| --- | --- | --- |
| `enabled` | Enable sandbox execution | `False` |
| `engine` | Execution engine: `docker`, `e2b` or `volcengine` | `docker` |
| `default_config` | Environment settings such as image and working directory; overrides benchmark defaults | `{}` |
| `manager_config` | Connection settings such as endpoint, credentials and concurrency limits | `{}` |
| `pool_size` | Size of the code execution pool | Follows `eval_batch_size` |

Pass a dictionary as shown above, or use `evalscope.config.SandboxTaskConfig`. For agents, benchmarks can also provide per-sample environment settings. See [native agent configuration](agent/native.md) for command timeouts and other agent parameters.

## Use E2B Remote Sandboxes

E2B supports the [official service](https://e2b.dev/) and [Aliyun Agent Sandbox](https://help.aliyun.com/zh/functioncompute/product-overview-of-fc-agent-sandbox). Docker is not required on the machine running the evaluation.

### Installation and Credentials

Requires Python ≥3.11 and an ms-enclave version with E2B support. The example below runs SWE-bench, so install its benchmark dependencies as well:

```bash
pip install -U 'evalscope[swe_bench]' 'ms-enclave[e2b]'
export DASHSCOPE_API_KEY='<your-model-api-key>'
export E2B_API_KEY='<your-sandbox-api-key>'
```

`DASHSCOPE_API_KEY` is used for model inference, while `E2B_API_KEY` is used to create sandboxes. The sandbox key must belong to your selected provider.

### Run an Evaluation

This example runs one SWE-bench Agentic sample on official E2B. The benchmark supplies the image, so no manual download is needed:

```python
import os

from evalscope import TaskConfig, run_task

manager_config = {'max_sandboxes': 2}

task_config = TaskConfig(
    model='qwen3.8-flash',
    eval_type='openai_api',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    datasets=['swe_bench_verified_agentic'],
    limit=1,
    eval_batch_size=1,
    agent_config={'mode': 'native', 'max_steps': 80},
    sandbox={
        'enabled': True,
        'engine': 'e2b',
        'default_config': {'sandbox_timeout': 3600},
        'manager_config': manager_config,
    },
)

run_task(task_config)
```

For Aliyun Agent Sandbox, replace `manager_config` in the example with:

```python
manager_config = {
    'api_url': 'https://api.cn-hangzhou.sandbox.aliyuncs.com',
    'domain': 'cn-hangzhou.sandbox.aliyuncs.com',
    'max_sandboxes': 2,
}
```

`api_url` and `domain` must refer to the same region. Set `E2B_API_KEY` to the E2B-compatible key issued by Aliyun. Legacy domains such as `<id>.<region>.e2b.fc.aliyuncs.com` are also supported.

### Supported Benchmarks

| Benchmark | Official E2B | Aliyun Agent Sandbox |
| --- | --- | --- |
| SWE-bench Verified, Verified Mini, Lite, and their Agentic variants | Supported | Supported |
| SWE-bench Multilingual Agentic, SWE-bench Pro | Supported | Supported |
| SWE-Atlas QnA | Supported | Supported |
| WideSearch, HumanEval | Supported | Requires an existing template |

Aliyun cannot build templates directly from Docker Hub. For benchmarks without an available image mapping, prepare a template and specify its name or ID through `sandbox.default_config.template`. For other code evaluation benchmarks, check their sandbox configuration for the required languages, tools and dependencies.

The following benchmarks do not currently support E2B: `gaia`, `officeqa`, `job_bench`, `gdpval`, `skillsbench` and `cybench`. Use Docker instead.

E2B currently supports native agents, but not `ExternalAgentConfig`. Terminal-Bench, DeepSWE and AutomationBench runs managed by Harbor/Pier use their own execution environments and are not switched through this `sandbox.engine` setting.

### Common Parameters

Place connection and concurrency settings in `sandbox.manager_config`:

| Parameter | Default | Usage |
| --- | --- | --- |
| `api_key` | `E2B_API_KEY` environment variable | Can also be supplied explicitly in the configuration |
| `api_url`, `domain` | Official E2B endpoints | Aliyun users must supply the endpoints for their region |
| `max_sandboxes` | `16` | Maximum number of live sandboxes; must fit your account quota |
| `max_concurrent_builds` | `4` | Maximum concurrent template builds when preparing multiple images |

Place environment settings in `sandbox.default_config`:

| Parameter | Default | Usage |
| --- | --- | --- |
| `image` | Supplied by the benchmark | Usually needs no change; custom images must include the required environment |
| `template` | Unset | Name or ID of an existing template |
| `sandbox_timeout` | `3600` seconds | Maximum sandbox lifetime; must fit your account quota |
| `cpu_count` | `2` | CPU count used when building a template |
| `memory_mb` | `4096` MiB | Memory used when building a template |
| `working_dir`, `env_vars` | Supplied by the benchmark | Command working directory and environment variables |

The first use of an image waits for its template to become ready. Subsequent runs can reuse the template. When using an existing template, its environment and resource specifications apply.

For concurrency and timeouts:

- The code execution `pool_size` must not exceed `max_sandboxes`; invalid settings produce a configuration error at startup.
- `sandbox_timeout` controls sandbox lifetime. Configure it separately from the agent's `command_timeout` and evaluation script timeout, allowing enough time for the task.
- Official Hobby accounts have a maximum sandbox lifetime of 3600 seconds. For SWE-bench Pro, explicitly set `sandbox_timeout=3600` as shown above. Limits for other plans and Aliyun depend on your account quota.

Docker settings such as volume mounts, port mappings and privileged mode cannot be used directly with E2B. Unsupported fields produce a configuration error. For further configuration options, see the [ms-enclave documentation](https://github.com/modelscope/ms-enclave).

### Troubleshooting and Cleanup

| Issue | Action |
| --- | --- |
| Missing key or authentication failure | Check `E2B_API_KEY` and confirm that it belongs to the selected provider |
| No available image mapping | Use an existing `template`, or prepare a suitable image mapping |
| Template build failure | Check the template name and provider error in the log, fix the image or account permissions, and rerun |
| Concurrency or lifetime quota exceeded | Reduce concurrency, shorten `sandbox_timeout`, or adjust the provider quota |
| Unsupported field or benchmark | Follow the configuration error or use Docker |

Sandbox preparation failures stop the evaluation with an error by default. Set `TaskConfig(ignore_errors=True)` explicitly to skip failed samples and continue. Failed samples are not scored; resuming from cache retries the failed step and reuses any model output already generated. Sandboxes are released on normal completion. After an unexpected exit, use the `run_id` from the log to find and clean up resources left by that run:

```bash
npx --yes @e2b/cli sandbox list --metadata 'run_id=<run-id>' --format json
npx --yes @e2b/cli sandbox kill <sandbox-id>
```

Use the same provider credentials as the evaluation. Aliyun also requires the corresponding `E2B_API_URL` and `E2B_DOMAIN`.

## Use Self-Hosted Remote Docker

Install and start Docker on the remote server, then install ms-enclave and start its service:

```bash
pip install 'ms-enclave[docker]'
ms-enclave server --host 0.0.0.0 --port 1234
```

Install `evalscope[sandbox]` locally; local Docker is not required. In the local Docker example, add the remote address to `sandbox.manager_config`:

```python
task_config.sandbox.manager_config = {'base_url': 'http://<remote_host>:1234'}
run_task(task_config)
```

## Use VolcEngine SandboxFusion

Start a [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started/) service:

```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

Use the following sandbox configuration in your evaluation task:

```python
sandbox = {
    'enabled': True,
    'engine': 'volcengine',
    'manager_config': {
        'base_url': 'http://<remote_host>:8080',
        'dataset_language_map': {
            'r': 'R',
            'd_ut': 'D_ut',
            'ts': 'typescript',
        },
    },
}
```
