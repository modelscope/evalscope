# Sandbox Execution

EvalScope uses [ms_enclave](https://github.com/modelscope/ms-enclave) to run
model-generated code inside isolated sandboxes.  Two execution paths share
the same service layer:

* **Pooled** — a warm pool of long-lived sandboxes used by code benchmarks
  (e.g. HumanEval, MBPP) through the `SandboxMixin`.
* **Per-sample** — one sandbox per sample created by agent environments
  (e.g. `EnclaveAgentEnvironment`) for SWE-bench-style benchmarks.

Both paths are backed by the process-wide `SandboxService` defined in
`evalscope.api.sandbox`, which caches `SandboxManager` instances keyed on
`(engine, manager_config)` and cleans them up at exit.

## Installation

```bash
pip install 'evalscope[sandbox]'
```

Requires a reachable Docker daemon or valid Volcengine credentials
(depending on the chosen engine).

## Configuration

Prefer the nested `sandbox` object on `TaskConfig`.  The legacy fields
`use_sandbox` / `sandbox_type` / `sandbox_manager_config` remain as
deprecated aliases and are kept in sync automatically.

### Python

```python
from evalscope import TaskConfig
from evalscope.config import SandboxTaskConfig

task = TaskConfig(
    datasets=['humaneval'],
    sandbox=SandboxTaskConfig(
        enabled=True,
        engine='docker',          # 'docker' | 'volcengine'
        default_config={
            'image': 'python:3.11-slim',
            'working_dir': '/workspace',
        },
        manager_config={
            # e.g. 'base_url': 'unix:///var/run/docker.sock'
        },
        pool_size=8,              # optional, defaults to eval_batch_size
    ),
)
```

### YAML / JSON

```yaml
sandbox:
  enabled: true
  engine: volcengine
  manager_config:
    api_key: ${VOLC_API_KEY}
    region: cn-beijing
  default_config:
    image: my-registry/my-image:latest
```

## Supported engines

| `engine`     | Aliases              | ms_enclave backend             |
|--------------|----------------------|--------------------------------|
| `docker`     | (default)            | `SandboxManagerFactory`        |
| `volcengine` | `volcano`, `volc`    | `VolcengineSandboxManager`     |

## Per-sample overrides (agent benchmarks)

Agent benchmarks that need a different sandbox per sample (e.g. SWE-bench
Pro, which pins an image per instance) can override `build_sandbox_config`
on their adapter:

```python
class MyBenchmarkAdapter(DefaultDataAdapter):
    def build_sandbox_config(self, sample):
        return {'image': sample.metadata['docker_image']}
```

The returned dict is merged on top of `sandbox.default_config` and passed
to the environment constructor.  `agent_config.environment_extra` still
has the final word if the user wants to override per-run.

## Pool vs per-sample

| Aspect           | Pool (SandboxMixin)                | Per-sample (Agent env)        |
|------------------|------------------------------------|-------------------------------|
| Lifetime         | Reused across samples              | Fresh container per sample    |
| Warmup           | `manager.initialize_pool`          | None                          |
| Execution API    | `PoolHandle.execute_tool`          | `SandboxHandle.execute_tool`  |
| Typical use      | Code benchmarks                    | SWE-bench, Agent tool use     |

Managers are shared across both paths when `(engine, manager_config)`
match, so you only pay for one connection even if a benchmark uses both.

## API reference

The public surface lives in `evalscope.api.sandbox`:

- `SandboxEngine`, `resolve_engine(value)`
- `SandboxService`, `get_sandbox_service()`
- `PoolHandle`, `SandboxHandle`
- `build_sandbox_config(engine, cfg_dict)`,
  `merge_sandbox_config_dicts(*dicts)`

See `tests/agent/test_sandbox_service.py` for runnable examples.
