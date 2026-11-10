# 沙箱环境使用

沙箱用于运行模型生成的代码，或为 Agent 提供独立的命令执行环境。EvalScope 通过 [ms-enclave](https://github.com/modelscope/ms-enclave) 创建和管理这些环境，评测结束后自动释放资源。

## 选择使用方式

| 使用方式 | 适用场景 | 准备工作 |
| --- | --- | --- |
| 本地 Docker | 在本机运行代码评测或 Agent | 本机安装并启动 Docker |
| 自建远端 Docker | 将执行任务交给自己的服务器 | 在服务器安装 Docker 并启动 ms-enclave 服务 |
| E2B 远端沙箱 | 使用官方 E2B 或阿里云 Agent Sandbox，无需自行维护 Docker 服务 | 准备服务商账号和 E2B 兼容 API key |
| VolcEngine SandboxFusion | 使用已有的 SandboxFusion 服务 | 准备可访问的服务地址 |

代码评测基准（如 [HumanEval](../benchmarks/humaneval.md)）通常已提供默认镜像；[Agent 评测](agent/index.md) 的环境要求以对应基准文档为准。

## 快速开始：本地 Docker

安装 [Docker](https://www.docker.com/get-started) 并启动服务，然后安装依赖：

```bash
pip install 'evalscope[sandbox]'
export DASHSCOPE_API_KEY='<your-model-api-key>'
```

以下示例使用 Docker 在 HumanEval 上评测 5 个样本：

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

首次运行会下载镜像。请根据机器的 CPU 和内存调整并发数；多语言镜像通常需要更多资源。

## 通用配置

通过 `TaskConfig.sandbox` 设置沙箱：

| 字段 | 作用 | 默认值 |
| --- | --- | --- |
| `enabled` | 是否启用沙箱 | `False` |
| `engine` | 引擎：`docker`、`e2b` 或 `volcengine` | `docker` |
| `default_config` | 镜像、工作目录等环境设置；覆盖基准的默认配置 | `{}` |
| `manager_config` | 服务地址、凭据、并发上限等连接设置 | `{}` |
| `pool_size` | 代码执行池的大小 | 跟随 `eval_batch_size` |

既可以传入上例中的字典，也可以使用 `evalscope.config.SandboxTaskConfig`。对于 Agent，基准可以进一步提供每个样本的环境配置；命令超时等 Agent 参数见 [原生 Agent 配置](agent/native.md)。

## 使用 E2B 远端沙箱

E2B 支持[官方服务](https://e2b.dev/)和[阿里云 Agent Sandbox](https://help.aliyun.com/zh/functioncompute/product-overview-of-fc-agent-sandbox)，运行评测的机器无需安装 Docker。

### 安装与凭据

需要 Python ≥3.11，以及已支持 E2B 的 ms-enclave 版本。以下示例运行 SWE-bench，因此同时安装对应基准依赖：

```bash
pip install -U 'evalscope[swe_bench]' 'ms-enclave[e2b]'
export DASHSCOPE_API_KEY='<your-model-api-key>'
export E2B_API_KEY='<your-sandbox-api-key>'
```

`DASHSCOPE_API_KEY` 用于模型推理，`E2B_API_KEY` 用于创建沙箱。后者必须属于你选择的沙箱服务商。

### 运行评测

下面使用官方 E2B 运行一个 SWE-bench Agentic 样本。镜像由基准提供，无需手动下载：

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

使用阿里云 Agent Sandbox 时，将上例中的 `manager_config` 替换为：

```python
manager_config = {
    'api_url': 'https://api.cn-hangzhou.sandbox.aliyuncs.com',
    'domain': 'cn-hangzhou.sandbox.aliyuncs.com',
    'max_sandboxes': 2,
}
```

`api_url` 与 `domain` 需要对应同一地域，`E2B_API_KEY` 使用阿里云提供的 E2B 兼容密钥。也支持旧版 `<id>.<region>.e2b.fc.aliyuncs.com` 域名。

### 支持的基准

| 基准 | 官方 E2B | 阿里云 Agent Sandbox |
| --- | --- | --- |
| SWE-bench Verified、Verified Mini、Lite，以及对应的 Agentic 版本 | 支持 | 支持 |
| SWE-bench Multilingual Agentic、SWE-bench Pro | 支持 | 支持 |
| SWE-Atlas QnA | 支持 | 支持 |
| WideSearch、HumanEval | 支持 | 需要指定已有模板 |

阿里云不能直接从 Docker Hub 构建模板。对于没有可用镜像映射的基准，请先准备模板，并通过 `sandbox.default_config.template` 指定模板名或 ID。其他代码执行基准所需的语言、工具和依赖，以各基准的沙箱配置为准。

以下基准暂不支持 E2B：`gaia`、`officeqa`、`job_bench`、`gdpval`、`skillsbench`、`cybench`。请选择 Docker。

E2B 目前支持原生 Agent，不支持 `ExternalAgentConfig`。通过 Harbor/Pier 运行的 Terminal-Bench、DeepSWE 和 AutomationBench 使用各自的执行环境，不通过这里的 `sandbox.engine` 切换。

### 常用参数

连接和并发设置放在 `sandbox.manager_config`：

| 参数 | 默认值 | 使用说明 |
| --- | --- | --- |
| `api_key` | 环境变量 `E2B_API_KEY` | 也可在配置中显式传入 |
| `api_url`、`domain` | 官方 E2B 端点 | 阿里云用户需要填写对应地域的端点 |
| `max_sandboxes` | `16` | 同时存活的沙箱数量上限，需符合账号配额 |
| `max_concurrent_builds` | `4` | 首次准备多个镜像时的模板构建并发上限 |

环境设置放在 `sandbox.default_config`：

| 参数 | 默认值 | 使用说明 |
| --- | --- | --- |
| `image` | 基准提供 | 通常无需修改；自定义镜像需包含基准所需环境 |
| `template` | 未设置 | 使用已有模板时填写模板名或 ID |
| `sandbox_timeout` | `3600` 秒 | 沙箱的最长存活时间，需符合账号配额 |
| `cpu_count` | `2` | 构建模板时使用的 CPU 数量 |
| `memory_mb` | `4096` MiB | 构建模板时使用的内存大小 |
| `working_dir`、`env_vars` | 基准提供 | 命令工作目录与环境变量 |

首次使用镜像时需要等待模板就绪，后续运行可复用模板。使用已有模板时，环境和资源规格由该模板决定。

并发与超时需要注意：

- 代码执行池的 `pool_size` 不能超过 `max_sandboxes`，否则会在启动时报告配置错误。
- `sandbox_timeout` 是沙箱存活时间，与 Agent 的 `command_timeout`、评测脚本超时分别设置；请为实际任务预留足够时间。
- 官方 Hobby 账号的沙箱存活上限为 3600 秒；运行 SWE-bench Pro 时，请像上例一样显式设置 `sandbox_timeout=3600`。其他套餐和阿里云的限制以账号配额为准。

Docker 的挂载、端口映射和特权模式等设置不能直接用于 E2B，遇到不支持的字段时会报配置错误。更多底层配置见 [ms-enclave 文档](https://github.com/modelscope/ms-enclave)。

### 常见问题与清理

| 问题 | 处理方式 |
| --- | --- |
| 密钥缺失或鉴权失败 | 检查 `E2B_API_KEY`，确认密钥属于当前服务商 |
| 镜像没有可用映射 | 使用已有 `template`，或准备符合要求的镜像映射 |
| 模板构建失败 | 根据日志中的模板名和平台错误检查镜像及账号权限，修正后重新运行 |
| 并发或存活时间超出配额 | 调低并发、缩短 `sandbox_timeout`，或调整服务商配额 |
| 不支持的字段或基准 | 按错误提示调整配置，或使用 Docker |

沙箱准备失败时不会调用模型，该样本计 0 分；恢复缓存运行时会重新尝试准备环境。正常结束会释放沙箱。异常退出后，可根据日志中的 `run_id` 查询和清理本次运行的残留：

```bash
npx --yes @e2b/cli sandbox list --metadata 'run_id=<run-id>' --format json
npx --yes @e2b/cli sandbox kill <sandbox-id>
```

清理时使用与运行时相同的服务商凭据；阿里云还需设置对应的 `E2B_API_URL` 和 `E2B_DOMAIN`。

## 使用自建远端 Docker

在远端服务器上安装并启动 Docker，然后安装 ms-enclave 并启动服务：

```bash
pip install 'ms-enclave[docker]'
ms-enclave server --host 0.0.0.0 --port 1234
```

本地安装 `evalscope[sandbox]`，无需安装 Docker。在“本地 Docker”示例中，为 `sandbox.manager_config` 增加远端地址即可：

```python
task_config.sandbox.manager_config = {'base_url': 'http://<remote_host>:1234'}
run_task(task_config)
```

## 使用 VolcEngine SandboxFusion

启动 [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started/) 服务：

```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

在评测任务中使用以下沙箱配置：

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
