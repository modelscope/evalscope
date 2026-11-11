# 沙箱环境使用

代码评测需要运行模型生成的程序，Agent 评测则可能需要执行命令、修改文件或运行测试。沙箱为这些操作提供独立的运行环境。EvalScope 通过 [ms-enclave](https://github.com/modelscope/ms-enclave) 管理沙箱，自动完成创建、执行和资源释放。

## 选择使用方式

| 使用方式 | 适用场景 | 准备工作 |
| --- | --- | --- |
| [本地 Docker](#local-docker) | 使用本机资源运行代码或 Agent | 本机安装并启动 Docker |
| [自建远端 Docker](#remote-docker) | 使用自己的服务器运行代码或 Agent | 在服务器启动 ms-enclave 服务 |
| [E2B 远端沙箱](#e2b) | 使用云端环境，无需本机 Docker 或自建服务器 | 准备官方 E2B 或阿里云的账号与沙箱 API key |
| [VolcEngine SandboxFusion](#sandboxfusion) | 连接已有的 SandboxFusion 服务 | 准备可访问的服务地址 |

以下示例需要 Python 3.11 或更高版本。代码评测基准（如 [HumanEval](../benchmarks/humaneval.md)）通常已配置好运行镜像；[Agent 评测](agent/index.md) 的环境要求请查阅对应基准文档。

(local-docker)=
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

首次运行时，EvalScope 会下载所需镜像并启动容器。`eval_batch_size=4` 表示最多同时处理 4 个样本，请根据本机 CPU 和内存调整。评测结果默认保存在 `outputs/` 下，目录结构见[输出文件说明](../get_started/parameters.md)。

(remote-docker)=
## 使用自建远端 Docker

如果本机资源不足，可以将沙箱部署在自己的服务器上。本机负责调用模型和组织评测，代码在服务器的 Docker 容器中运行。

在服务器上安装并启动 Docker，然后启动 ms-enclave 服务：

```bash
pip install 'ms-enclave[docker]'
ms-enclave server --host 0.0.0.0 --port 1234
```

本机安装 `evalscope[sandbox]`，无需安装 Docker。沿用本地 Docker 示例，在调用 `run_task(task_config)` **之前**设置服务地址：

```python
task_config.sandbox.manager_config = {'base_url': 'http://<remote_host>:1234'}
run_task(task_config)
```

(e2b)=
## 使用 E2B 远端沙箱

如果不想自行部署 Docker，可以选择[官方 E2B](https://e2b.dev/) 或[阿里云 Agent Sandbox](https://help.aliyun.com/zh/functioncompute/product-overview-of-fc-agent-sandbox)。两者都通过 `e2b` 引擎接入，代码和命令在云端执行，本机只需安装评测依赖。

### 安装依赖并准备密钥

下面以 SWE-bench 为例。PyPI 上的 EvalScope 1.14.1 尚未包含 E2B 接入，请安装 `main` 分支的源码版本；这条命令会同时安装 SWE-bench 和沙箱依赖：

```bash
pip install -U 'evalscope[swe_bench] @ git+https://github.com/modelscope/evalscope.git@main'
export DASHSCOPE_API_KEY='<your-model-api-key>'
export E2B_API_KEY='<your-sandbox-api-key>'
```

这里需要两份密钥：`DASHSCOPE_API_KEY` 用于调用示例中的模型，`E2B_API_KEY` 用于创建沙箱。使用官方 E2B 时填写官方密钥，使用阿里云时填写阿里云提供的 E2B 兼容密钥。

如果运行其他代码评测基准，可将安装命令中的 `[swe_bench]` 换成 `[sandbox]`；无需再单独安装 `ms-enclave[e2b]`。

### 运行评测

以下脚本使用官方 E2B 评测一个 SWE-bench Agentic 样本。模型会在沙箱中查看仓库、修改代码并提交补丁，EvalScope 随后运行测试评分。基准已指定运行镜像，无需手动下载。

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

使用阿里云时，在创建 `TaskConfig` **之前**将脚本中的 `manager_config` 定义替换为：

```python
manager_config = {
    'api_url': 'https://api.cn-hangzhou.sandbox.aliyuncs.com',
    'domain': 'cn-hangzhou.sandbox.aliyuncs.com',
    'max_sandboxes': 2,
}
```

上例使用杭州地域。切换地域时，请同时修改 `api_url` 和 `domain`，并确认 `E2B_API_KEY` 属于当前服务商。这里也兼容阿里云早期的 `*.e2b.fc.aliyuncs.com` 域名格式；新配置使用上面的 `sandbox.aliyuncs.com` 地址即可。

### 服务地址与兼容性

`manager_config.api_url` 用于创建、查询和释放沙箱，`domain` 用于访问沙箱内的命令和文件服务。显式配置优先于环境变量 `E2B_API_URL`、`E2B_DOMAIN`，均未设置时使用 SDK 默认端点。沙箱密钥优先读取 `manager_config.api_key`，其次读取 `E2B_API_KEY`。

其他 E2B 兼容服务也可填写自己的地址和密钥，但需要兼容 SDK 的沙箱创建、命令执行、文件读写和释放接口；自动构建镜像还需要模板接口支持。目前已验证官方 E2B 和阿里云，其他服务需确认兼容性。服务商的地址识别规则见下方参数参考。

### 镜像与模板

模板保存沙箱启动时的运行环境。首次使用某个镜像时，EvalScope 会自动构建模板；后续可复用模板，创建新的独立沙箱。评测结束后释放沙箱，模板保留供后续使用。

SWE-bench 单镜像实测：新建模板约需 10–22 秒，复用后约 1–2 秒即可执行命令（中位数），具体耗时因镜像和网络而异。

阿里云不能直接从 Docker Hub 构建模板。EvalScope 为下表中的 SWE-bench 和 SWE-Atlas 镜像提供了内置映射，会自动选择预先准备的阿里云镜像，不会现场转换或上传。其他非阿里云地址不使用这张映射表。

没有映射时，请先准备包含基准依赖的模板，再将模板名或 ID 填入 `sandbox.default_config.template`。指定模板后会跳过镜像映射和构建，环境及资源规格以该模板为准。

### 支持的基准

| 基准 | 官方 E2B | 阿里云 Agent Sandbox |
| --- | --- | --- |
| SWE-bench Verified、Verified Mini、Lite，以及对应的 Agentic 版本 | 支持 | 支持 |
| [SWE-bench Multilingual Agentic](../benchmarks/swe_bench_multilingual_agentic.md)、[SWE-bench Pro](../benchmarks/swe_bench_pro.md) | 支持 | 支持 |
| [SWE-Atlas QnA](../benchmarks/swe_atlas_qna.md) | 支持 | 支持 |
| [WideSearch](../benchmarks/wide_search.md)、[HumanEval](../benchmarks/humaneval.md) | 支持 | 需指定包含相应依赖的模板 |

其他代码评测基准能否使用 E2B，取决于镜像中的语言、工具和依赖是否满足要求，请查阅对应基准文档。

以下基准暂不支持 E2B：`gaia`、`officeqa`、`job_bench`、`gdpval`、`skillsbench`、`cybench`。请选择 Docker。

通过本页 `sandbox` 配置运行 Agent 时，目前仅支持[原生 Agent](agent/native.md)，不支持[外部 Agent](agent/bridge.md)。以下基准的运行环境另行配置：

- [Terminal-Bench](../third_party/terminal_bench.md) 通过 Harbor 运行。安装 `harbor[e2b]` 并设置 `E2B_API_KEY` 后，可在 `dataset_args.<基准名>.extra_params` 中设置 `environment_type='e2b'` 来选择 Harbor 的 E2B 后端。该后端不使用本页的 `sandbox.manager_config` 和镜像映射，具体任务的资源要求见对应基准文档。
- [DeepSWE](../benchmarks/deep_swe.md) 当前适配器使用 Pier 的 Docker 环境。
- [AutomationBench](../benchmarks/automation_bench.md) 使用官方 Python 运行时，不经过本页的沙箱引擎。

### 调整并发和运行时长

`eval_batch_size` 控制同时处理的样本数，`sandbox.manager_config.max_sandboxes` 限制同时存在的沙箱数。增加并发前，请确认服务商账号有足够配额。HumanEval 等使用代码执行池的基准，还需保证 `sandbox.pool_size` 不超过 `max_sandboxes`；未设置 `pool_size` 时，取 `eval_batch_size`。

`sandbox.default_config.sandbox_timeout` 控制沙箱从创建到回收的最长时长，默认 3600 秒。它与 Agent 单条命令的 `command_timeout`、基准测试脚本的超时不同，需要为整个任务预留足够时间。

[官方 E2B Hobby 账号](https://e2b.dev/docs/sandbox)的存活时长上限为 3600 秒，运行 SWE-bench Pro 时也请保留示例中的 `sandbox_timeout=3600`。其他套餐和阿里云的限制以账号配额为准。

```{dropdown} E2B 参数参考
服务商根据地址识别，不根据密钥识别：`api_url` 或 `domain` 的主机名以 `.aliyuncs.com` 结尾时，启用阿里云镜像映射；其他地址使用默认兼容模式。阿里云服务若使用自定义代理域名，可能无法被这条规则识别。

服务连接与并发参数放在 `sandbox.manager_config`：

| 参数 | 默认值 | 使用说明 |
| --- | --- | --- |
| `api_key` | 环境变量 `E2B_API_KEY` | 也可显式传入，保存的配置快照会隐藏密钥 |
| `api_url`、`domain` | 对应环境变量，未设置时使用 SDK 默认值 | 阿里云用户需要填写对应地域的端点；其他兼容服务使用各自端点 |
| `max_sandboxes` | `16` | 同时存活的沙箱数量上限，需符合账号配额 |
| `max_concurrent_builds` | `4` | 首次准备多个镜像时的模板构建并发上限 |
| `build_mode` | 自动选择 | 阿里云默认 `micro`（MicroVM），官方 E2B 使用 `container`；`micro` 只能用于阿里云 |
| `image_region` | 根据服务地址推导 | 覆盖阿里云镜像地址中的地域，如 `cn-shanghai`；目标地域需已有对应镜像，不会自动复制 |
| `image_mapping` | EvalScope 内置映射表 | 自定义原镜像到阿里云镜像的对应关系；显式传入时替换整张内置表，不是追加 |

环境设置放在 `sandbox.default_config`：

| 参数 | 默认值 | 使用说明 |
| --- | --- | --- |
| `image` | 基准提供 | 通常无需修改；自定义镜像需包含基准所需环境 |
| `template` | 未设置 | 使用已有模板时填写模板名或 ID |
| `sandbox_timeout` | `3600` 秒 | 沙箱的最长存活时间，需符合账号配额 |
| `cpu_count` | `2` | 构建模板时使用的 CPU 数量 |
| `memory_mb` | `4096` MiB | 构建模板时使用的内存大小 |
| `disk_size_mb` | `15360` MiB | 构建模板时的磁盘大小，阿里云 `micro` 模式下生效 |
| `user` | `root` | 执行命令的用户 |
| `working_dir` | 基准提供 | 命令的工作目录 |
| `env_vars` | `{}` | 为命令添加环境变量 |

SWE-bench 在评分时会另建沙箱，默认存活时长为测试脚本超时加 600 秒；显式设置 `sandbox_timeout` 后，取两者中的较小值。测试脚本默认超时为 SWE-bench 1800 秒、SWE-bench Pro 3600 秒。

Docker 的目录挂载、端口映射和特权模式不能直接用于 E2B，不支持的配置会报错。更多参数见 [ms-enclave 文档](https://github.com/modelscope/ms-enclave)。
```

### 常见问题与清理

| 问题 | 处理方式 |
| --- | --- |
| 密钥缺失或鉴权失败 | 检查 `E2B_API_KEY`，确认密钥属于当前服务商 |
| 镜像没有可用映射 | 准备包含基准依赖的模板，并设置 `sandbox.default_config.template` |
| 模板构建失败 | 根据日志中的模板名和平台错误检查镜像及账号权限，修正后重新运行 |
| 并发或存活时间超出配额 | 调低并发、缩短 `sandbox_timeout`，或调整服务商配额 |
| 不支持的字段或基准 | 按错误提示调整配置，或使用 Docker |

沙箱准备失败时，评测默认报错并停止。只有显式设置 `TaskConfig(ignore_errors=True)` 才会跳过失败样本并继续，失败样本不计入分数。修复问题后，将 `use_cache` 指向原评测输出目录，即可复用已有结果并重试未完成的步骤，详见[缓存参数](../get_started/parameters.md)。

正常结束时，EvalScope 会释放沙箱。如果程序意外退出，可用日志中的 `run_id` 查找本次运行留下的沙箱，再按 ID 清理。先设置与评测时相同的 `E2B_API_KEY`；阿里云还需设置对应的 `E2B_API_URL` 和 `E2B_DOMAIN`：

```bash
npx --yes @e2b/cli sandbox list --metadata 'run_id=<run-id>' --format json
npx --yes @e2b/cli sandbox kill '<sandbox-id>'
```

(sandboxfusion)=
## 使用 VolcEngine SandboxFusion

如果已有 [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started/) 服务，可以直接连接；也可以在服务器上用 Docker 启动：

```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

本机安装 `evalscope[sandbox]`。沿用本地 Docker 示例，在调用 `run_task(task_config)` **之前**切换引擎并设置服务地址：

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

`base_url` 填写实际服务地址。`dataset_language_map` 是可选配置，用于指定数据集名称与 SandboxFusion 语言标识的对应关系；上例列出了三个映射。

## 通用配置参考

各类沙箱都通过 `TaskConfig.sandbox` 配置。可以传入上述示例中的字典，也可以使用 `evalscope.config.SandboxTaskConfig`。

| 字段 | 作用 | 默认值 |
| --- | --- | --- |
| `enabled` | 是否启用沙箱 | `False` |
| `engine` | 执行引擎：`docker`、`e2b` 或 `volcengine` | `docker` |
| `default_config` | 镜像、工作目录等环境设置，覆盖基准默认值 | `{}` |
| `manager_config` | 服务地址、凭据和并发上限等设置 | `{}` |
| `pool_size` | 代码执行池的沙箱数量；不控制 Agent 的样本并发 | 跟随 `eval_batch_size` |

Agent 的命令超时、工具和步数等参数通过 `agent_config` 设置，详见[原生 Agent 配置](agent/native.md)。
