# Sandbox Environments

Code evaluations run model-generated programs, while agent evaluations may execute commands, edit files or run tests. Sandboxes provide an isolated environment for these operations. EvalScope uses [ms-enclave](https://github.com/modelscope/ms-enclave) to create these environments, execute tasks and release resources automatically.

## Choose an Execution Environment

| Option | Use case | Prerequisites |
| --- | --- | --- |
| [Local Docker](#local-docker) | Run code or agents using your machine's resources | Install and start Docker locally |
| [Self-hosted remote Docker](#remote-docker) | Run code or agents on your own server | Start the ms-enclave service on the server |
| [E2B remote sandboxes](#e2b) | Use cloud environments without local Docker or your own server | An official E2B or Aliyun account and a sandbox API key |
| [VolcEngine SandboxFusion](#sandboxfusion) | Connect to an existing SandboxFusion service | An accessible service endpoint |

The examples below require Python 3.11 or later. Code evaluation benchmarks such as [HumanEval](../benchmarks/humaneval.md) typically provide a default image. For [agent evaluations](agent/index.md), check the benchmark documentation for environment requirements.

(local-docker)=
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

On the first run, EvalScope downloads the required image and starts the containers. `eval_batch_size=4` allows up to four samples to run concurrently; adjust it to fit your machine's CPU and memory. Results are saved under `outputs/` by default. See the [output directory reference](../get_started/parameters.md) for its layout.

(remote-docker)=
## Use Self-Hosted Remote Docker

To use your own server's resources, deploy the sandbox service there. Your local machine calls the model and coordinates the evaluation, while code runs in Docker containers on the server.

Install and start Docker on the server, then start the ms-enclave service:

```bash
pip install 'ms-enclave[docker]'
ms-enclave server --host 0.0.0.0 --port 1234
```

Install `evalscope[sandbox]` locally; local Docker is not required. Reuse the local Docker example, setting the service address **before** calling `run_task(task_config)`:

```python
task_config.sandbox.manager_config = {'base_url': 'http://<remote_host>:1234'}
run_task(task_config)
```

(e2b)=
## Use E2B Remote Sandboxes

To run without deploying Docker, choose [official E2B](https://e2b.dev/) or [Aliyun Agent Sandbox](https://help.aliyun.com/zh/functioncompute/product-overview-of-fc-agent-sandbox). Both use the `e2b` engine. Code and commands run in the cloud; your local machine only needs the evaluation dependencies.

### Install Dependencies and Set API Keys

This example uses SWE-bench. Install the evaluation and sandbox dependencies:

```bash
pip install -U 'evalscope[swe_bench]'
export DASHSCOPE_API_KEY='<your-model-api-key>'
export E2B_API_KEY='<your-sandbox-api-key>'
```

The example uses DashScope and reads its model API key from `DASHSCOPE_API_KEY`. You can use any OpenAI-compatible service by replacing `model`, `api_url` and `api_key` below. Set `E2B_API_KEY` to the sandbox API key issued by your selected sandbox provider.

For other code evaluation benchmarks, replace `[swe_bench]` with `[sandbox]` in the installation command. You do not need to install `ms-enclave[e2b]` separately.

### Run an Evaluation

The following script evaluates one SWE-bench Agentic sample on official E2B. The model explores the repository, edits code and submits a patch in the sandbox; EvalScope then runs the tests to grade it. The benchmark supplies the image, so no manual download is needed.

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

For Aliyun, replace the script's `manager_config` definition **before** creating `TaskConfig`:

```python
manager_config = {
    'api_url': 'https://api.cn-hangzhou.sandbox.aliyuncs.com',
    'domain': 'cn-hangzhou.sandbox.aliyuncs.com',
    'max_sandboxes': 2,
}
```

This configuration uses Hangzhou. To change regions, update both `api_url` and `domain`, and make sure `E2B_API_KEY` belongs to the selected provider.

### Endpoints and Compatibility

`manager_config.api_url` is used to create, query and remove sandboxes; `domain` is used to reach command and file services inside them. Explicit settings take precedence over `E2B_API_URL` and `E2B_DOMAIN`; when neither is set, SDK defaults apply. Sandbox credentials are read from `manager_config.api_key` first, then `E2B_API_KEY`.

You can also supply another E2B-compatible service's endpoints and key. It must support the SDK interfaces for creating sandboxes, executing commands, reading and writing files, and releasing resources. Automatic image builds also require template support. Official E2B and Aliyun have been verified; compatibility with other services needs to be confirmed. The provider detection rule is described in the parameter reference below.

### Images and Templates

A template stores the environment a sandbox starts with. When an image is first used, EvalScope builds a template automatically. Later runs can reuse it to create new, isolated sandboxes. Sandboxes are released after evaluation; templates are retained for reuse.

In a test with one SWE-bench image, preparing a new template took about 10–22 seconds, and a new sandbox using an existing template could execute a command in about 1–2 seconds (median); timings vary with the image and network.

Aliyun cannot build templates directly from Docker Hub. EvalScope includes mappings for the SWE-bench and SWE-Atlas images listed below and automatically selects prepared Aliyun images, without converting or uploading images at runtime. Non-Aliyun endpoints do not use this mapping table.

For custom images without a mapping, prepare a template with the benchmark's dependencies and set `sandbox.default_config.template` to its name or ID. Specifying a template skips image mapping and builds; its environment and resource specifications apply.

### Supported Benchmarks

| Benchmark | Official E2B | Aliyun Agent Sandbox |
| --- | --- | --- |
| SWE-bench Verified, Verified Mini, Lite, and their Agentic variants | Supported | Supported |
| [SWE-bench Multilingual Agentic](../benchmarks/swe_bench_multilingual_agentic.md), [SWE-bench Pro](../benchmarks/swe_bench_pro.md) | Supported | Supported |
| [SWE-Atlas QnA](../benchmarks/swe_atlas_qna.md) | Supported | Supported |
| [WideSearch](../benchmarks/wide_search.md) | Supported | Supported; selects the `base` template automatically (Python 3.13) |
| [HumanEval](../benchmarks/humaneval.md), [MBPP](../benchmarks/mbpp.md), [LiveCodeBench](../benchmarks/live_code_bench.md) | Supported | Supported; selects the `base` template automatically (Python 3.13) |
| [HumanEvalPlus](../benchmarks/humaneval_plus.md), [MBPPPlus](../benchmarks/mbpp_plus.md) | Requires an image or template with NumPy | Requires a template with NumPy |
| [BigCodeBench](../benchmarks/bigcodebench.md), [BigCodeBench-Hard](../benchmarks/bigcodebench_hard.md) | Supported | Requires a suitable evaluation template |
| [DS-1000](../benchmarks/ds1000.md), [SciCode](../benchmarks/scicode.md) | Requires an image or template with the dependencies and test data | Requires a suitable evaluation template |

For custom environments, prepare dependencies, test data and working directories as described in each benchmark's documentation. WideSearch also needs network access to search services and target websites. Use [native agent mode](agent/native.md) for agent evaluations.

[Terminal-Bench 2.0/2.1](../third_party/terminal_bench.md) can also use E2B through Harbor: install `harbor[e2b]` and set `environment_type='e2b'` in the benchmark's `extra_params`.

### Set Concurrency and Sandbox Lifetime

`eval_batch_size` controls how many samples run concurrently, while `sandbox.manager_config.max_sandboxes` limits the number of live sandboxes. Check your provider quota before increasing concurrency. For benchmarks that use a code execution pool, such as HumanEval, `sandbox.pool_size` must not exceed `max_sandboxes`. If omitted, `pool_size` follows `eval_batch_size`.

`sandbox.default_config.sandbox_timeout` sets the maximum time from sandbox creation to removal, defaulting to 3600 seconds. This differs from the agent's per-command `command_timeout` and the benchmark test script timeout. Allow enough time for the whole task.

[Official E2B Hobby accounts](https://e2b.dev/docs/sandbox) have a maximum sandbox lifetime of 3600 seconds. Keep `sandbox_timeout=3600` from the example when running SWE-bench Pro as well. Limits for other plans and Aliyun depend on your account quota.

### E2B Parameter Reference

Provider handling is determined by the endpoint, not the key. If the hostname in either `api_url` or `domain` ends in `.aliyuncs.com`, Aliyun image mapping is enabled. Other endpoints use the default compatibility path. An Aliyun service behind a custom proxy domain may be missed by this rule.

Place connection and concurrency settings in `sandbox.manager_config`:

| Parameter | Default | Usage |
| --- | --- | --- |
| `api_key` | `E2B_API_KEY` environment variable | Can also be supplied explicitly; saved configuration snapshots hide the key |
| `api_url`, `domain` | Corresponding environment variables, then SDK defaults | Use matching regional endpoints for Aliyun, or the endpoints of another compatible service |
| `max_sandboxes` | `16` | Maximum number of live sandboxes; must fit your account quota |
| `max_concurrent_builds` | `4` | Maximum concurrent template builds when preparing multiple images |
| `build_mode` | Automatic | `micro` (MicroVM) on Aliyun and `container` on official E2B; `micro` works only on Aliyun |
| `image_region` | Inferred from the endpoint | Overrides the region in Aliyun image addresses, such as `cn-shanghai`; the image must already exist there, as images are not copied automatically |
| `image_mapping` | EvalScope's bundled mapping | Custom source-to-Aliyun image mappings; an explicit dictionary replaces the entire bundled mapping rather than extending it |

Place environment settings in `sandbox.default_config`:

| Parameter | Default | Usage |
| --- | --- | --- |
| `image` | Supplied by the benchmark | Usually needs no change; custom images must include the required environment |
| `template` | Unset | Name or ID of an existing template |
| `sandbox_timeout` | `3600` seconds | Maximum sandbox lifetime; must fit your account quota |
| `cpu_count` | `2` | CPU count used when building a template |
| `memory_mb` | `4096` MiB | Memory used when building a template |
| `disk_size_mb` | `15360` MiB | Disk size used when building a template; applies to Aliyun `micro` mode |
| `user` | `root` | User that runs commands |
| `working_dir` | Supplied by the benchmark | Command working directory |
| `env_vars` | `{}` | Environment variables to add to commands |

SWE-bench creates a separate sandbox for grading. Its default lifetime is the test script timeout plus 600 seconds; if `sandbox_timeout` is explicitly set, the smaller value applies. The default test script timeout is 1800 seconds for SWE-bench and 3600 seconds for SWE-bench Pro.

Docker volume mounts, port mappings and privileged mode cannot be used directly with E2B; unsupported settings produce an error. For more options, see the [ms-enclave documentation](https://github.com/modelscope/ms-enclave).

### Troubleshooting and Cleanup

| Issue | Action |
| --- | --- |
| Missing key or authentication failure | Check `E2B_API_KEY` and confirm that it belongs to the selected provider |
| No available image mapping | Prepare a template with the benchmark's dependencies and set `sandbox.default_config.template` |
| Template build failure | Check the template name and provider error in the log, fix the image or account permissions, and rerun |
| Concurrency or lifetime quota exceeded | Reduce concurrency, shorten `sandbox_timeout`, or adjust the provider quota |
| Unsupported field or benchmark | Follow the configuration error or use Docker |

Sandbox preparation failures stop the evaluation with an error by default. Only an explicit `TaskConfig(ignore_errors=True)` skips failed samples and continues; failed samples are not scored. After fixing the problem, set `use_cache` to the original output directory to reuse completed results and retry unfinished steps. See the [cache parameters](../get_started/parameters.md).

EvalScope releases sandboxes on normal completion. If the process exits unexpectedly, use the `run_id` from the log to find sandboxes left by that run, then remove them by ID. First set the same `E2B_API_KEY` used for the evaluation; Aliyun also requires the corresponding `E2B_API_URL` and `E2B_DOMAIN`:

```bash
npx --yes @e2b/cli sandbox list --metadata 'run_id=<run-id>' --format json
npx --yes @e2b/cli sandbox kill '<sandbox-id>'
```

(sandboxfusion)=
## Use VolcEngine SandboxFusion

Connect to an existing [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started/) service, or start one with Docker on a server:

```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

Install `evalscope[sandbox]` locally. Reuse the local Docker example, changing the engine and setting the service address **before** calling `run_task(task_config)`:

```python
task_config.sandbox.engine = 'volcengine'
task_config.sandbox.manager_config = {
    'base_url': 'http://<remote_host>:8080',
    'dataset_language_map': {
        'r': 'R',
        'd_ut': 'D_ut',
        'ts': 'typescript',
    },
}
run_task(task_config)
```

Set `base_url` to your service address. The optional `dataset_language_map` maps dataset names to SandboxFusion language identifiers; the example lists three mappings.

## Shared Configuration Reference

All sandbox types are configured through `TaskConfig.sandbox`. Pass a dictionary as shown above, or use `evalscope.config.SandboxTaskConfig`.

| Field | Purpose | Default |
| --- | --- | --- |
| `enabled` | Enable sandbox execution | `False` |
| `engine` | Execution engine: `docker`, `e2b` or `volcengine` | `docker` |
| `default_config` | Environment settings such as image and working directory; overrides benchmark defaults | `{}` |
| `manager_config` | Connection settings such as endpoint, credentials and concurrency limits | `{}` |
| `pool_size` | Number of sandboxes in the code execution pool; does not control agent sample concurrency | Follows `eval_batch_size` |

Agent command timeouts, tools and step limits are configured through `agent_config`. See [native agent configuration](agent/native.md).
