# Toolathlon

Toolathlon 用于评测智能体在真实 MCP 软件环境中的长程工具使用能力。EvalScope 的 `toolathlon` 集成是官方 Toolathlon evaluation service 的 wrapper，不重新实现 MCP 环境、任务容器、agent loop 或官方评分器。

## 安装

```bash
pip install evalscope[toolathlon]
```

EvalScope wrapper 只需要客户端依赖 `httpx` 和 `websockets`。如果使用官方公共服务，不需要安装完整 Toolathlon 仓库。

## 服务模式

EvalScope 支持 Toolathlon 官方服务的 private mode。

在 private mode 中，Toolathlon 服务负责运行任务容器、MCP 环境、agent loop 和评分。EvalScope 会启动本地 WebSocket relay；当服务端需要模型响应时，请求会通过 WebSocket 转回 EvalScope 所在机器，再调用你的本地或内网 OpenAI-compatible endpoint。因此即使 Toolathlon 服务在远端，也可以使用 `api_url=http://localhost:8000/v1`。

官方 Toolathlon client 还支持 public mode，即服务端直接用提交的 API key 调用公共模型 API。EvalScope wrapper 目前固定使用 private mode，让模型 endpoint 和 API key 留在 EvalScope 侧。

## 官方公共服务

默认服务是官方公共服务：

- HTTP service: `47.253.6.47:8080`
- WebSocket proxy: `47.253.6.47:8081`

官方公共服务主要用于快速试跑和调试。根据官方文档，它按 IP 在 24 小时窗口内限流：

- 每个 IP 每 24 小时累计执行时长 180 分钟
- 每个 IP 每 24 小时 3 次 evaluation request
- 累计执行时长低于 180 分钟时，不受 3 次请求数限制
- 累计执行时长超过 180 分钟后，开始应用 3 次请求数限制

如果提交阶段返回 `HTTP 503 Service Unavailable`，通常表示官方服务忙，因为服务端一次只运行一个 evaluation job。如果返回 `HTTP 429`，则表示触发了 IP 限流。

## 快速调试

建议先用很小的 `task_list` 调试：

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

`skip_container_restart=True` 只建议用于小规模调试。正式评测时应移除它，让官方服务在任务之间重启容器。

## 自建官方服务

自建服务可以避开公共服务排队和公共 IP 限流，但它不是一个 `docker compose up` 就能完成的部署。官方 Docker 镜像 `docker.io/lockon0927/toolathlon-task-image:1016beta` 主要是任务运行镜像，不包含 Toolathlon evaluation service 所需的所有服务、账号、凭据和评测进程。

启动自建服务前，先按官方 Toolathlon 材料完成环境准备：

- `README.md`：安装依赖、Docker/Podman、`configs/global_configs.py`、应用凭据、本地服务部署和 smoke check
- `global_preparation/how2register_accounts.md`：MCP 任务需要的账号、token 和 session
- `configs/global_configs_example.py`：复制为 `configs/global_configs.py`，并配置 `podman_or_docker`
- `global_preparation/pull_toolathlon_image.sh`：拉取官方任务镜像
- `global_preparation/deploy_containers.sh`：部署 Canvas、email、WooCommerce、k8s/kind 等本地服务
- `EVAL_SERVICE_README.md`：启动官方 `eval_server.py` HTTP 服务和 private mode WebSocket proxy

完整 Toolathlon 环境准备好后，在 Toolathlon 仓库中启动官方服务：

```bash
python eval_server.py 8080 8081 3 10 180
```

参数含义：

- `server_port`：HTTP 服务端口
- `ws_proxy_port`：private mode 的 WebSocket proxy 端口
- `max_submissions_per_ip`：每个 IP 每 24 小时请求数限制，`-1` 表示不限
- `max_workers`：每个 job 内部最大的 Toolathlon workers 数
- `max_duration_minutes`：每个 IP 每 24 小时累计执行时长限制，`-1` 表示不限

如果是内部自用服务，可以关闭限流：

```bash
python eval_server.py 8080 8081 -1 10 -1
```

## 在 EvalScope 中使用自建服务

通过 `extra_params` 指向自建服务：

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

如果使用 CLI，把同样的配置放到 `--dataset-args` 的 `toolathlon.extra_params` 下即可。

## 运维注意点

- 官方服务一次只运行一个 evaluation job。`workers` 控制的是单个 job 内部并行度，不是提交 job 的并发数。
- 官方服务为了驱动 agent loop 和评分，会接收 prompt、模型响应和工具上下文。
- 需要保持 client protocol 和官方服务版本一致。当前 EvalScope wrapper 对齐 Toolathlon service protocol `1.3`。
- 长期或正式评测建议使用 dedicated official service 或自建服务，不建议依赖共享公共服务。

官方来源：

- [Toolathlon repository](https://github.com/hkust-nlp/Toolathlon)
- [Toolathlon evaluation service README](https://github.com/hkust-nlp/Toolathlon/blob/main/EVAL_SERVICE_README.md)
