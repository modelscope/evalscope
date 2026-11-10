# 沙箱环境使用

为了完成LLM代码能力评测，我们需要搭建一套独立的评测环境，避免在开发环境执行错误代码从而造成不可避免的损失。目前，EvalScope 接入了[ms-enclave](https://github.com/modelscope/ms-enclave) 沙箱环境，允许用户在受控的环境中评测模型的代码能力，例如使用HumanEval、LiveCodeBench等评测基准。

接下来介绍两种不同的沙箱使用方式：

- 本地使用：在本地机器上搭建沙箱环境，并在本地进行评测，需本地机器支持Docker；
- 异地使用：在远程服务器上搭建沙箱环境，并通过API接口进行评测，需远端机器支持Docker。

```{tip}
- 数据集默认的**沙箱环境配置**可以在数据集配置中查看，例如[HumanEval](../benchmarks/humaneval.md)。
- **沙箱工作进程数** 需要根据本地机器的资源情况进行调整，建议设置为不超过本地CPU核心数的一半，例如本地有8核CPU，则建议设置为4，尤其是多语言镜像(`volcengine/sandbox-fusion`)，其消耗资源更多，建议适当减少工作进程数。
- 同一套沙箱基础设施同时驱动 [Agent 评测](agent/index.md) 中的 `agent_config.environment='docker'`，`image` / `timeout` 等参数放在 `agent_config.environment_extra`。
```

## 0. 统一 Sandbox 配置

EvalScope 通过嵌套字段 `TaskConfig.sandbox`（对应 `SandboxTaskConfig`）来统一管理沙箱设置。

`SandboxTaskConfig` 字段：

| 字段 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `enabled` | `bool` | 是否启用沙箱 | `false` |
| `engine` | `str` | 沙箱引擎：`docker` / `volcengine` 等 | `docker` |
| `default_config` | `dict` | 任务级沙箱配置，会与 `BenchmarkMeta.sandbox_config` 合并；同时作为 Agent 模式中每个样本环境的默认配置 | `{}` |
| `manager_config` | `dict` | 转发给 ms_enclave manager 的参数（远端 docker 的 `base_url`、volcengine 凭证等） | `{}` |
| `pool_size` | `int \| None` | 池化执行的预热池大小，`None` 时跟随 `eval_batch_size` | `None` |

`sandbox` 字段同时接受 `SandboxTaskConfig` 实例与等价字典，两种写法行为一致：

```python
from evalscope.config import SandboxTaskConfig

# 方式 A：使用 SandboxTaskConfig（推荐，享有类型提示）
TaskConfig(
    sandbox=SandboxTaskConfig(
        enabled=True,
        engine='docker',
        manager_config={'base_url': 'http://remote:1234'},
    ),
)

# 方式 B：直接传 dict
TaskConfig(
    sandbox={
        'enabled': True,
        'engine': 'docker',
        'manager_config': {'base_url': 'http://remote:1234'},
    },
)
```

## 1. 本地使用

使用Docker在本地机器上搭建沙箱环境，并在本地进行评测，需本地机器支持Docker

### 安装环境

1. **安装Docker**：请确保您的机器上已安装Docker。您可以从[Docker官网](https://www.docker.com/get-started)下载并安装Docker。

2. **安装沙箱环境相关依赖**：在本地使用的Python环境中安装`ms-enclave`等包：

```bash
pip install evalscope[sandbox]
```

### 配置参数
在运行评测时，通过 `sandbox` 字段启用沙箱环境，其余参数与普通评测相同：

下面是一个在HumanEval上进行模型评测的完整示例代码：
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
        'enabled': True,    # 启用沙箱
        'engine': 'docker', # 指定沙箱引擎
    },
)

run_task(task_config)
```

在模型评测过程中，EvalScope会自动启动并管理沙箱环境，确保代码在隔离的环境中运行。控制台有如下输出：
```text
[INFO:ms_enclave] Local sandbox manager started
...
```

## 2. 异地使用

在远程服务器上搭建沙箱环境，并通过API接口进行评测，需远端机器支持Docker。

### 安装环境

你需要分别在远端机器和本地机器上进行安装和配置。

#### 远端机器

在远端机器上安装的环境与上述本地使用方式类似：

1. **安装Docker**：请确保您的机器上已安装Docker。您可以从[Docker官网](https://www.docker.com/get-started)下载并安装Docker。

2. **安装沙箱环境相关依赖**：在远端机器的Python环境中安装`ms-enclave`等包：

```bash
pip install evalscope[sandbox]
```

3. **启动沙箱服务器**：运行以下命令以启动沙箱服务器：

```bash
ms-enclave server --host 0.0.0.0 --port 1234
```

#### 本地机器

此时本地机器无需安装Docker，但需要安装EvalScope：

```bash
pip install evalscope[sandbox]
```

### 配置参数

在运行评测时，通过 `sandbox` 字段启用沙箱环境，并在 `manager_config` 中指定远端沙箱服务器的API地址：

完整示例代码如下：
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
        'enabled': True,    # 启用沙箱
        'engine': 'docker', # 指定沙箱引擎
        'manager_config': {
            'base_url': 'http://<remote_host>:1234'  # 远端沙箱管理器URL
        },
    },
)

run_task(task_config)
```

在模型评测过程中，EvalScope会通过API与远端沙箱服务器进行通信，确保代码在隔离的环境中运行。控制台有如下输出：
```text
[INFO:ms_enclave] HTTP sandbox manager started, connected to http://<remote_host>:1234
...
```

## 3. 额外支持的沙箱环境

### 使用 E2B 远端沙箱

E2B 引擎在远端创建独立沙箱，不需要本机 Docker。需要 Python 3.11 或更高版本，以及带 E2B 支持的 ms-enclave：

```bash
pip install -U 'ms-enclave[e2b]'
export E2B_API_KEY='<your-e2b-compatible-api-key>'
```

运行前准备服务商账号、E2B API key、并发和存活时长配额。阿里云用户还需要同一地域的 `api_url` 与 `domain`。
API key 默认读取 `E2B_API_KEY`，也可传入 `manager_config.api_key`；配置快照中的任意层级 `api_key` 都会脱敏。
镜像转换表随 EvalScope 自动加载，用户无需配置。不要把脱敏快照中的占位符作为真实密钥使用。

e2b 官方示例（模型服务密钥与沙箱密钥分别设置）：

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

阿里云 Agent Sandbox 示例：

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

上例的 `E2B_API_KEY` 必须属于所选服务商；阿里云同样使用 E2B 兼容 API key，不需要 Team ID 或 AK/SK。
旧域名 `<id>.<region>.e2b.fc.aliyuncs.com` 也可用。

#### 配置项

| manager_config 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `api_key` | 环境变量 `E2B_API_KEY` | 必填凭据；输出脱敏 |
| `api_url` | SDK 默认端点 | 阿里云为 `https://api.<region>.sandbox.aliyuncs.com` |
| `domain` | SDK 默认域名 | 阿里云为 `<region>.sandbox.aliyuncs.com` |
| `max_sandboxes` | 16 | 当前管理器同时存活沙箱上限；正整数 |
| `max_concurrent_builds` | 4 | 同时构建并等待就绪的模板上限；正整数 |
| `build_mode` | 阿里云 `micro`，官方 `container` | 官方不支持 `micro` |

| default_config 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `image` | benchmark 提供 | 官方源镜像；阿里云自动转换为对应 ACR 镜像 |
| `template` | 未设置 | 指定已有模板时，跳过镜像映射、地域改写和构建 |
| `sandbox_timeout` | 3600 秒 | 创建时固定存活时长，不续期、不 pause |
| `cpu_count` | 2 | 正整数；参与模板构建与命名 |
| `memory_mb` | 4096 MiB | 正整数；参与模板构建与命名 |
| `disk_size_mb` | 15360 MiB | 正整数；仅 `micro` 构建通过请求头生效，始终参与模板名哈希 |
| `user` | `root` | EvalScope 统一默认用户，避免 /testbed 的 git 所有权问题 |
| `working_dir` | benchmark 提供，SDK 配置默认 /home/user | 命令工作目录 |
| `env_vars` | `{}` | 每条命令的环境变量 |
| `tools_config` | Agent 环境默认 shell_executor、python_executor | ms-enclave 通用工具配置 |
| `allow_internet_access` | true | 创建时允许访问互联网 |
| `timeout` | 30 秒 | 继承字段；不控制沙箱存活时长 |
| `resource_limits` | `{}` | 继承字段；E2B 不使用 |

已有 Docker 字段中，`memory_limit` 向上换算为 MiB、`cpu_limit` 向上取整、`network_enabled` 转为联网开关；
显式 E2B 字段优先。容器 command、platform、拉取日志、退出移除、线程数与 cap_add 会被丢弃。
非默认的 volumes、privileged、entrypoint、ports、extra_hosts、network 会在运行前报 `SandboxConfigError`，未知字段同样报错。

#### 模板、配额与存活时长

池化代码执行的 `sandbox.pool_size`（未设置时取 `eval_batch_size`）不能超过 `max_sandboxes`，
否则启动时报配置错误，以免预热池等待永远不会释放的名额。

首次使用镜像会构建模板，相同镜像与构建资源配置复用模板。同一管理器内构建失败会缓存失败，重新运行后可重试；
远端已有 error 状态的自动模板会删除重建一次。显式模板不会被自动删除。

阿里云默认使用 `micro`，附带 MicroVM 和磁盘大小请求头；可显式选择 `container`。
官方使用 `container`，不附带厂商请求头。

实测官方 Hobby 沙箱存活上限为 3600 秒，阿里云为 86400 秒，以账号当前配额为准。
SWE-bench 与 Pro 的**评测沙箱**存活时长为
`min(评测超时 + 600 秒, 用户显式设置的 sandbox_timeout)`；未显式设置时为评测超时加 600 秒。
推理沙箱默认仍为 3600 秒。官方 Hobby 运行 `swe_bench_pro` 时必须显式设 `sandbox_timeout=3600`，
否则默认评测时长 3600 加余量会超出套餐上限。平台先回收或评测超时都按未解决计分。
沙箱准备失败时不调用模型，样本计 0 分；恢复缓存运行会重试这些样本。

`swe_bench_pro` 的评测版本为 `v1.1`：评测沙箱会收到数据集的 `test_patch`，
存为上游脚本要求的 `/tests/test_patch.patch`，修复旧版本缺少测试补丁的问题。
因此 Pro 的旧版本分数与 `v1.1` 不应直接混用；其他 benchmark 的评测版本保持不变。

#### 高级：镜像地域

| manager_config 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `image_region` | 从端点推导 | 覆盖 ACR 镜像的目标地域 |

默认支持新旧阿里云端点自动推导，例如 cn-hangzhou 和 ap-southeast-1。
`image_region` 只改写 ACR 主机中的地域，不改变镜像路径或 tag。

#### 兼容性

下表区分实现范围和真实服务验收；“可用未验证”不代表已经通过端到端验收。
“已验证”指本期每个 benchmark 的单样本 gold/空 patch 冒烟，Agent 链路另做小样本验证，不代表全量数据验收。
Mini 和 Multilingual 还在两家服务上完成了真实模型补丁提交及官方评分；SWE-Atlas 完成了真实模型答案和全部 5 项 rubric 评分。
这些单样本运行提高了 Agent 步数并明确提交约束，部分需要模型修正格式或继续提交；评分完成不等于模型成功解题。
Pro 使用上游固定版本的脚本与解析器；若解析出的测试名与数据集要求不一致，仍按原始名称比较并判为未解决。

| benchmark | e2b 官方 | 阿里云 |
| --- | --- | --- |
| swe_bench_verified | 已验证 | 已验证 |
| swe_bench_verified_mini | 已验证 | 已验证 |
| swe_bench_lite | 已验证 | 已验证 |
| swe_bench_verified_agentic | 已验证 | 已验证 |
| swe_bench_verified_mini_agentic | 已验证 | 已验证 |
| swe_bench_lite_agentic | 已验证 | 已验证 |
| swe_bench_multilingual_agentic | 已验证 | 已验证 |
| swe_bench_pro | 已验证 | 已验证 |
| swe_atlas_qna | 单样本答案及 judge 已验证 | 单样本答案及 judge 已验证 |
| wide_search | 可用未验证 | 需补镜像映射 |
| 池化代码执行 benchmark（如 HumanEval） | 可用未验证 | 需补镜像映射 |

阿里云不能直接从 docker.io 构建模板，wide_search 和池化代码执行的默认镜像尚不在内置映射内；
可以使用已准备的 template。以下六项第一期选择 E2B 会报 `SandboxConfigError`：

| benchmark | 状态 | 原因与后续改造难度 |
| --- | --- | --- |
| gaia | e2b 不支持（第一期） | 只读挂载附件目录；改为上传样本附件，难度小 |
| officeqa | e2b 不支持（第一期） | 只读挂载本地语料；上传或预装模板，难度中 |
| job_bench | e2b 不支持（第一期） | 读写挂载输出目录、本机评分；需要下载产物，难度中 |
| gdpval | e2b 不支持（第一期） | 本机 Dockerfile 构建、挂载参考资料；需远端构建和上传，难度大 |
| skillsbench | e2b 不支持（第一期） | 本机任务镜像、联网任务访问宿主服务；需远端构建和网络改造，难度大 |
| cybench | e2b 不支持（第一期） | privileged、Docker-in-Docker、cgroup、本机 inspect；难度大 |

`ExternalAgentConfig` 与 E2B 的组合不支持。Harbor/Pier 驱动的 terminal_bench、deep_swe、automation_bench
不经过此通用引擎，本期不支持。Docker 仍为默认引擎。

#### 残留清理与排错

正常结束会 kill 沙箱；异常退出时平台到期回收只作兜底。启动日志的 metadata `run_id` 可用于定位本次残留。
使用与运行时相同的服务商环境变量（阿里云需额外设置 E2B_API_URL、E2B_DOMAIN）：

```bash
npx --yes @e2b/cli sandbox list --metadata 'run_id=<run-id>' --format json
npx --yes @e2b/cli sandbox kill <sandbox-id>
# 仅清理指定运行，不要省略 metadata 过滤条件：
npx --yes @e2b/cli sandbox kill --all --metadata 'run_id=<run-id>'
```

- 缺 api_key：设置 E2B_API_KEY，核对密钥所属服务商。
- 缺 ACR 映射：检查源镜像，补充映射资源或指定已有 template。
- 模板构建失败：查看模板名、源/目标镜像及平台原始原因；修改后重新运行。
- 配额不足：降低 max_sandboxes 或 max_concurrent_builds，核对账号配额。
- 超过存活上限：调小 sandbox_timeout 或升级套餐，错误会附带实际值与 Hobby 上限提示。
- 不支持的字段/benchmark：按错误列出的字段修改配置或使用 Docker。


### 使用VolcEngine沙箱环境

EvalScope 还支持使用[VolcEngine 沙箱环境](https://bytedance.github.io/SandboxFusion/docs/docs/get-started/) ，用户可以选择使用VolcEngine提供的沙箱镜像来进行代码评测。

1. **安装Docker**：请确保您的机器上已安装Docker。您可以从[Docker官网](https://www.docker.com/get-started)下载并安装Docker。
2. **启动VolcEngine沙箱服务器**：
启动VolcEngine沙箱服务器的命令如下：

```bash
docker run -it -p 8080:8080 vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609
```

3. **配置评测参数**：
使用VolcEngine沙箱环境时，只需将`engine`参数设置为`volcengine`，并确保远端机器上已安装并运行了VolcEngine沙箱服务器。

```python
    ...
    sandbox={
        'enabled': True,        # 启用沙箱
        'engine': 'volcengine', # 指定使用VolcEngine沙箱
        'manager_config': {
            'base_url': 'http://<remote_host>:8080',  # 远端VolcEngine沙箱管理器URL
            'dataset_language_map': {  # 可选，指定数据集对应的编程语言
                'r': 'R',
                'd_ut': 'D_ut',
                'ts': 'typescript',
            },
        },
    },
    ...
```

控制台输出如下：
```text
2026-01-20 08:07:22 [debug    ] running command python /tmp/tmpyhpyii_k/tmpf_q4hcoq.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpv1hp4llu/tmpowprm7uz.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpeytkvisz/tmpxkproid8.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpyhpyii_k/tmpf_q4hcoq.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmpqcbuep0x/tmplxz4z9cw.py [sandbox.runners.base]
2026-01-20 08:07:23 [debug    ] stop running command python /tmp/tmp3msgd871/tmplldfzrp9.py [sandbox.runners.base]
...
```
