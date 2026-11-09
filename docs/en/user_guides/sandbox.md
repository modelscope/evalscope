# Sandbox Environment Usage

To complete LLM code capability evaluation, we need to set up an independent evaluation environment to avoid executing erroneous code in the development environment and causing unavoidable losses. Currently, EvalScope has integrated the [ms-enclave](https://github.com/modelscope/ms-enclave) sandbox environment, allowing users to evaluate model code capabilities in a controlled environment, such as using evaluation benchmarks like HumanEval and LiveCodeBench.

The following introduces two different sandbox usage methods:

- Local usage: Set up the sandbox environment on a local machine and conduct evaluation locally, requiring Docker support on the local machine;
- Remote usage: Set up the sandbox environment on a remote server and conduct evaluation through API interfaces, requiring Docker support on the remote machine.

```{tip}
- The default **sandbox environment configuration** for datasets can be viewed in the dataset configuration, such as [HumanEval](../benchmarks/humaneval.md).
- The **number of sandbox worker processes** needs to be adjusted according to the resource situation of the local machine. It is recommended to set it to no more than half of the local CPU cores. For example, if the local machine has 8 CPU cores, it is recommended to set it to 4. Especially for multi-language images (`volcengine/sandbox-fusion`), which consume more resources, it is recommended to appropriately reduce the number of worker processes.
- The same sandbox infrastructure also powers `agent_config.environment='docker'` in [Agent Evaluation](agent/index.md); fields such as `image` / `timeout` belong in `agent_config.environment_extra`.
```

## 0. Unified Sandbox Configuration

EvalScope manages sandbox settings via the nested field `TaskConfig.sandbox` (mapped to `SandboxTaskConfig`).

`SandboxTaskConfig` fields:

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `enabled` | `bool` | Whether to enable the sandbox | `false` |
| `engine` | `str` | Sandbox engine: `docker` / `volcengine`, etc. | `docker` |
| `default_config` | `dict` | Task-level sandbox config; merged with `BenchmarkMeta.sandbox_config`, and used as the default per-sample environment config in Agent mode | `{}` |
| `manager_config` | `dict` | Forwarded to the ms_enclave manager (e.g. `base_url` for remote docker, volcengine credentials) | `{}` |
| `pool_size` | `int \| None` | Warmup pool size for pooled execution; falls back to `eval_batch_size` when `None` | `None` |

`sandbox` accepts both a `SandboxTaskConfig` instance and an equivalent dict — the two forms behave identically:

```python
from evalscope.config import SandboxTaskConfig

# Option A: use SandboxTaskConfig (recommended; type hints available)
TaskConfig(
    sandbox=SandboxTaskConfig(
        enabled=True,
        engine='docker',
        manager_config={'base_url': 'http://remote:1234'},
    ),
)

# Option B: pass a plain dict
TaskConfig(
    sandbox={
        'enabled': True,
        'engine': 'docker',
        'manager_config': {'base_url': 'http://remote:1234'},
    },
)
```

## 1. Local Usage

Use Docker to set up a sandbox environment on a local machine and conduct evaluation locally, requiring Docker support on the local machine.

### Environment Setup

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in your local Python environment:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration
Use the `sandbox` field on `TaskConfig` to enable the sandbox; other parameters remain the same as regular evaluations:

Here's a complete example code for model evaluation on HumanEval:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    model='qwen-plus',
    datasets=['humaneval'],
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    eval_batch_size=5,
    limit=5,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
        'seed': 42,
    },
    sandbox={
        'enabled': True,    # enable sandbox
        'engine': 'docker', # specify sandbox engine
    },
)

run_task(task_config)
```

During model evaluation, EvalScope will automatically start and manage the sandbox environment, ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. Remote Usage

Set up the sandbox environment on a remote server and conduct evaluation through API interfaces, requiring Docker support on the remote machine.

### Environment Setup

You need to install and configure separately on both the remote machine and local machine.

#### Remote Machine

The environment installation on the remote machine is similar to the local usage method described above:

1. **Install Docker**: Please ensure Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).

2. **Install sandbox environment dependencies**: Install packages like `ms-enclave` in remote Python environment:

```bash
pip install evalscope[sandbox]
```

3. **Start sandbox server**: Run the following command to start the sandbox server:

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### Local Machine

The local machine does not need Docker installation at this point, but needs to install EvalScope:

```bash
pip install evalscope[sandbox]
```

### Parameter Configuration

Use the `sandbox` field on `TaskConfig` to enable the sandbox, and specify the remote sandbox server's API address in `manager_config`:

Complete example code is as follows:
```python
from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    model='qwen-plus',
    datasets=['humaneval'],
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=env.get('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    eval_batch_size=5,
    limit=5,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
        'seed': 42,
    },
    sandbox={
        'enabled': True,    # enable sandbox
        'engine': 'docker', # specify sandbox engine
        'manager_config': {
            'base_url': 'http://<remote_host>:1234'  # remote sandbox manager URL
        },
    },
)

run_task(task_config)
```

During model evaluation, EvalScope will communicate with the remote sandbox server through API, ensuring code runs in an isolated environment. The console will display output like:
```text
[INFO:ms_enclave] HTTP sandbox manager started, connected to http://<remote_host>:1234
...
```

## 3. Additional Supported Sandbox

### Using E2B Remote Sandboxes

The E2B engine creates isolated remote sandboxes without a local Docker daemon.
Use Python 3.11+ and an ms-enclave version with E2B support:

```bash
pip install -U 'ms-enclave[e2b]'
export E2B_API_KEY='<your-e2b-compatible-api-key>'
```

Prepare a provider account, E2B-compatible API key, concurrency quota and lifetime quota.
Aliyun also requires an API URL and domain for the same region.
The key defaults to E2B_API_KEY; manager_config.api_key overrides it.
API keys at every depth of persisted sandbox settings are masked. The bundled image conversion table is loaded automatically.
Supply a real key when resuming; a masked placeholder is not a credential.

Official E2B example (model and sandbox credentials are separate):

```python
import os
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='qwen3.8-flash',
    eval_type='openai_api',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    datasets=['swe_bench_verified_agentic'],
    limit=1,
    agent_config={'mode': 'native', 'max_steps': 20},
    sandbox={
        'enabled': True,
        'engine': 'e2b',
        'default_config': {'sandbox_timeout': 3600},
        'manager_config': {'max_sandboxes': 2},
    },
))
```

Aliyun Agent Sandbox example:

```python
import os
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='qwen3.8-flash',
    eval_type='openai_api',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    datasets=['swe_bench_verified_agentic'],
    limit=1,
    agent_config={'mode': 'native', 'max_steps': 20},
    sandbox={
        'enabled': True,
        'engine': 'e2b',
        'default_config': {'sandbox_timeout': 3600},
        'manager_config': {
            'api_url': 'https://api.cn-hangzhou.sandbox.aliyuncs.com',
            'domain': 'cn-hangzhou.sandbox.aliyuncs.com',
            'max_sandboxes': 4,
        },
    },
))
```

E2B_API_KEY must belong to the selected provider. Aliyun uses its E2B-compatible API key, without Team ID or AK/SK.
Legacy domains such as `<id>.<region>.e2b.fc.aliyuncs.com` are also supported.

#### Configuration

| manager_config field | Default | Meaning |
| --- | --- | --- |
| api_key | E2B_API_KEY environment variable | Required credential; masked in outputs |
| api_url | SDK endpoint | Aliyun: https://api.&lt;region&gt;.sandbox.aliyuncs.com |
| domain | SDK domain | Aliyun: &lt;region&gt;.sandbox.aliyuncs.com |
| max_sandboxes | 16 | Positive integer; maximum live sandboxes in this manager |
| max_concurrent_builds | 4 | Positive integer; maximum builds including readiness polling |
| build_mode | Aliyun: micro; official: container | micro is supported only by Aliyun |

| default_config field | Default | Meaning |
| --- | --- | --- |
| image | Supplied by benchmark | Official source image, converted to ACR automatically on Aliyun |
| template | Unset | Existing template; bypasses mapping, region rewriting and builds |
| sandbox_timeout | 3600 seconds | Fixed lifetime at creation; no renewal or pause |
| cpu_count | 2 | Positive integer; included in template resources and name |
| memory_mb | 4096 MiB | Positive integer; included in template resources and name |
| disk_size_mb | 15360 MiB | Positive integer; applied through headers in micro mode, always hashed into the name |
| user | root | EvalScope default to avoid git ownership errors in /testbed |
| working_dir | Benchmark-defined; SDK config default /home/user | Command working directory |
| env_vars | {} | Environment variables passed to every command |
| tools_config | Agent environments: shell_executor, python_executor | Standard ms-enclave tool configuration |
| allow_internet_access | true | Internet access at creation |
| timeout | 30 seconds | Inherited field; does not control sandbox lifetime |
| resource_limits | {} | Inherited field; unused by E2B |

Docker memory_limit is rounded up to MiB, cpu_limit to an integer CPU count, and network_enabled to the internet flag.
Explicit E2B fields take precedence. Container command, platform, pull logging, removal, executor worker count and cap_add are dropped.
Nondefault volumes, privileged, entrypoint, ports, extra_hosts and network fail locally with SandboxConfigError.
Unknown fields also fail before remote preparation.

#### Templates, quotas and lifetime

For pooled code execution, sandbox.pool_size (default: eval_batch_size) must not exceed max_sandboxes.
An invalid combination fails locally rather than waiting indefinitely for warm-pool slots.

The first request builds a template. Identical images and build resources reuse it.
Concurrent requests share the build; a failed build is cached within the manager, so rerun after fixing the cause.
An existing automatic template in error state is deleted and rebuilt once. Explicit templates are never automatically deleted.

Aliyun defaults to micro with MicroVM and disk-size headers; container can be selected explicitly.
Official E2B uses container without provider-specific headers.

Observed lifetime limits are 3600 seconds for official Hobby accounts and 86400 seconds on Aliyun; check your account's current quota.
SWE-bench and Pro **evaluation sandbox** lifetime is
`min(evaluation timeout + 600 seconds, explicitly configured sandbox_timeout)`.
Without an explicit limit it is evaluation timeout plus 600 seconds. Inference still defaults to 3600 seconds.
Official Hobby users must set sandbox_timeout=3600 for swe_bench_pro, whose default evaluation timeout is already 3600 seconds.
Platform reclamation and evaluation timeouts fail closed as unresolved.
Preparation failures skip model calls and count as zero; cache resume retries those samples.

`swe_bench_pro` uses evaluation version `v1.1`: the evaluation sandbox receives the dataset's
`test_patch` at `/tests/test_patch.patch`, as required by the upstream script.
This fixes missing test-patch staging in the previous version. Keep Pro v1.1 scores separate from earlier scores;
other benchmark evaluation versions are unchanged.

#### Advanced: image region

| manager_config field | Default | Meaning |
| --- | --- | --- |
| image_region | Inferred from endpoint | Override the target ACR image region |

New and legacy Aliyun endpoints support automatic inference, including cn-hangzhou and ap-southeast-1.
Only the ACR registry host region changes; repository paths and tags remain unchanged.

#### Compatibility

This table distinguishes implementation scope from real provider acceptance.
“Available, unverified” does not claim end-to-end validation.
“Verified” means one gold/empty-patch sample per benchmark, with separate small native-agent runs; it is not full-dataset acceptance.
Pro preserves the pinned upstream scripts and parser: test names that do not match the dataset's required names remain unresolved.

| Benchmark | Official E2B | Aliyun |
| --- | --- | --- |
| swe_bench_verified | Verified | Verified |
| swe_bench_verified_mini | Verified | Verified |
| swe_bench_lite | Verified | Verified |
| swe_bench_verified_agentic | Verified | Verified |
| swe_bench_verified_mini_agentic | Verified | Verified |
| swe_bench_lite_agentic | Verified | Verified |
| swe_bench_multilingual_agentic | Verified | Verified |
| swe_bench_pro | Verified | Verified |
| swe_atlas_qna | Available, unverified | Available, unverified |
| wide_search | Available, unverified | Image mapping required |
| Pooled code benchmarks, e.g. HumanEval | Available, unverified | Image mapping required |

Aliyun cannot build directly from docker.io. Default wide_search and pooled code images are not in the bundled table;
an existing template can be supplied. Selecting E2B for any of the following raises SandboxConfigError in phase 1:

| Benchmark | Status | Reason and future migration effort |
| --- | --- | --- |
| gaia | E2B unsupported in phase 1 | Read-only attachment mounts; upload per-sample files, low effort |
| officeqa | E2B unsupported in phase 1 | Local corpus mounts; upload or preinstall a template, medium effort |
| job_bench | E2B unsupported in phase 1 | Writable output mounts and local scoring; requires artifact download, medium effort |
| gdpval | E2B unsupported in phase 1 | Local Dockerfile builds and reference mounts; remote builds and upload, high effort |
| skillsbench | E2B unsupported in phase 1 | Local task images and host services; remote builds and network changes, high effort |
| cybench | E2B unsupported in phase 1 | privileged, Docker-in-Docker, cgroups and local inspect, high effort |

ExternalAgentConfig with E2B is unsupported.
Harbor/Pier-driven terminal_bench, deep_swe and automation_bench do not use this engine and are outside phase 1.
Docker remains the default engine.

#### Cleanup and troubleshooting

Normal shutdown kills sandboxes. Platform expiry is only a fallback for unexpected process termination.
Use metadata run_id from the startup log to find leftovers.
Use the same provider environment variables as the evaluation; Aliyun also needs E2B_API_URL and E2B_DOMAIN:

```bash
npx --yes @e2b/cli sandbox list --metadata 'run_id=<run-id>' --format json
npx --yes @e2b/cli sandbox kill <sandbox-id>
# Restrict cleanup to the intended run; keep the metadata filter:
npx --yes @e2b/cli sandbox kill --all --metadata 'run_id=<run-id>'
```

- Missing api_key: set E2B_API_KEY and check which provider issued it.
- Missing ACR mapping: check the source image, extend the bundled mapping or supply a template.
- Template build failure: inspect the template name, source/target images and original provider error; fix and rerun.
- Quota exceeded: lower max_sandboxes or max_concurrent_builds and check account limits.
- Lifetime exceeded: lower sandbox_timeout or upgrade the plan; errors include the value and Hobby limit.
- Unsupported fields or benchmarks: follow the listed fields or use Docker.


### Using VolcEngine Sandbox Environment

EvalScope also supports using the [VolcEngine Sandbox Environment](https://bytedance.github.io/SandboxFusion/docs/docs/get-started/). Users can choose to use the sandbox images provided by VolcEngine for code evaluation.

1. **Install Docker**: Please ensure that Docker is installed on your machine. You can download and install Docker from the [Docker official website](https://www.docker.com/get-started).
2. **Start VolcEngine Sandbox Server**:
Use the following command to start the VolcEngine sandbox server:

```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

3. **Configure Evaluation Parameters**:
When using the VolcEngine sandbox environment, set the `engine` parameter to `volcengine`, and ensure that the VolcEngine sandbox server has been installed and is running on the remote machine.

```python
    ...
    sandbox={
        'enabled': True,        # Enable sandbox
        'engine': 'volcengine', # Use the VolcEngine sandbox
        'manager_config': {
            'base_url': 'http://<remote_host>:8080',  # Remote VolcEngine sandbox manager URL
            'dataset_language_map': {  # Optional, specify programming languages for datasets
                'r': 'R',
                'd_ut': 'D_ut',
                'ts': 'typescript',
            },
        },
    },
    ...
```

Console output will be as follows:
```text
2026-01-20 08:07:22 [debug    ] running command python /tmp/tmpyhpyii_k/tmpf_q4hcoq.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpv1hp4llu/tmpowprm7uz.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpeytkvisz/tmpxkproid8.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpyhpyii_k/tmpf_q4hcoq.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpqcbuep0x/tmplxz4z9cw.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmp3msgd871/tmplldfzrp9.py [sandbox.runners.base]
...
```
