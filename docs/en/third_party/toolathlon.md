# Toolathlon

Toolathlon evaluates long-horizon agent tool use in realistic MCP-backed software environments. EvalScope integrates it as a wrapper around the official Toolathlon evaluation service. EvalScope does not reimplement the MCP environments, task containers, agent loop, or official scorer.

## Install

```bash
pip install evalscope[toolathlon]
```

The EvalScope wrapper only needs the client-side dependencies `httpx` and `websockets`. You do not need to install the full Toolathlon repository when you use the official public service.

## Service Modes

EvalScope supports Toolathlon official service private mode.

In private mode, the Toolathlon service runs the task containers, MCP environments, agent loop, and scoring. EvalScope starts a local WebSocket relay; when the service needs a model response, the request is relayed back to your local or intranet OpenAI-compatible endpoint. This allows `api_url=http://localhost:8000/v1` to work even when the Toolathlon service is remote.

The official Toolathlon client also supports public mode, where the service directly calls a public model API with the submitted API key. The EvalScope wrapper intentionally uses private mode so the model endpoint and API key stay on the EvalScope side.

## Official Public Service

The default service is the official public service:

- HTTP service: `47.253.6.47:8080`
- WebSocket proxy: `47.253.6.47:8081`

The public service is intended for quick trials and debugging. According to the official documentation, it applies IP-based limits over a 24-hour window:

- 180 minutes cumulative execution time per IP per 24 hours
- 3 evaluation requests per IP per 24 hours
- While cumulative execution time is under 180 minutes, requests are not capped by the 3-request limit
- After cumulative execution time exceeds 180 minutes, the 3-request-per-24-hour cap applies

If submission returns `HTTP 503 Service Unavailable`, the service is usually busy because the official server runs one evaluation job at a time. If submission returns `HTTP 429`, the IP-based rate limit was reached.

## Quick Debug Run

Use a small `task_list` first:

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='your-model-name',
    api_url='http://localhost:8000/v1',
    api_key='your-local-api-key',
    datasets=['toolathlon'],
    dataset_args={
        'toolathlon': {
            'extra_params': {
                'task_list': ['find-alita-paper'],
                'skip_container_restart': True,
            }
        }
    },
    limit=1,
)

run_task(task_cfg)
```

Use `skip_container_restart=True` only for small debugging subsets. For formal runs, remove it so the official service can restart task containers between jobs.

## Self-hosted Official Service

Self-hosting removes the public-service queue and public IP limits, but it is not a single `docker compose up` deployment. The official Docker image `docker.io/lockon0927/toolathlon-task-image:1016beta` is mainly the task/runtime image; it does not package every service, account, credential, and evaluator process required by the Toolathlon evaluation service.

Before starting a self-hosted service, follow the official Toolathlon materials:

- `README.md`: installation dependencies, Docker/Podman setup, `configs/global_configs.py`, app credential setup, local app deployment, and local smoke checks
- `global_preparation/how2register_accounts.md`: accounts, tokens, and sessions required by MCP-backed tasks
- `configs/global_configs_example.py`: copy to `configs/global_configs.py` and set `podman_or_docker`
- `global_preparation/pull_toolathlon_image.sh`: pull the official task image
- `global_preparation/deploy_containers.sh`: deploy local services such as Canvas, email, WooCommerce, and k8s/kind
- `EVAL_SERVICE_README.md`: start the official `eval_server.py` HTTP service and private-mode WebSocket proxy

After the full Toolathlon environment is ready, start the official service from the Toolathlon repository:

```bash
python eval_server.py 8080 8081 3 10 180
```

The arguments are:

- `server_port`: HTTP service port
- `ws_proxy_port`: WebSocket proxy port for private mode
- `max_submissions_per_ip`: request limit per IP per 24 hours, or `-1` for unlimited
- `max_workers`: maximum Toolathlon workers per job
- `max_duration_minutes`: cumulative execution-time limit per IP per 24 hours, or `-1` for unlimited

For an internal service without rate limiting:

```bash
python eval_server.py 8080 8081 -1 10 -1
```

## Use a Self-hosted Service from EvalScope

Point EvalScope to the self-hosted service through `extra_params`:

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='your-model-name',
    api_url='http://localhost:8000/v1',
    api_key='your-local-api-key',
    datasets=['toolathlon'],
    dataset_args={
        'toolathlon': {
            'extra_params': {
                'server_host': 'your.toolathlon.server',
                'server_port': 8080,
                'ws_proxy_port': 8081,
                'workers': 10,
                'task_list': ['find-alita-paper'],
            }
        }
    },
    limit=1,
)

run_task(task_cfg)
```

For CLI usage, put the same values under `toolathlon.extra_params` in `--dataset-args`.

## Operational Notes

- The official server runs one evaluation job at a time. Increasing `workers` controls parallelism inside a job, not the number of concurrent submitted jobs.
- The official service receives prompts, model responses, and tool context in order to drive the agent loop and scoring.
- Keep `client_version` and WebSocket client protocol aligned with the official service version. This EvalScope wrapper follows Toolathlon service protocol `1.3`.
- For sustained evaluation, prefer a dedicated official service or a self-hosted service over the shared public endpoint.

Official sources:

- [Toolathlon repository](https://github.com/hkust-nlp/Toolathlon)
- [Toolathlon evaluation service README](https://github.com/hkust-nlp/Toolathlon/blob/main/EVAL_SERVICE_README.md)
