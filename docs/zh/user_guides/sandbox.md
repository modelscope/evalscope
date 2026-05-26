# 沙箱环境使用

为了完成LLM代码能力评测，我们需要搭建一套独立的评测环境，避免在开发环境执行错误代码从而造成不可避免的损失。目前，EvalScope 接入了[ms-enclave](https://github.com/modelscope/ms-enclave) 沙箱环境，允许用户在受控的环境中评测模型的代码能力，例如使用HumanEval、LiveCodeBench等评测基准。

接下来介绍两种不同的沙箱使用方式：

- 本地使用：在本地机器上搭建沙箱环境，并在本地进行评测，需本地机器支持Docker；
- 异地使用：在远程服务器上搭建沙箱环境，并通过API接口进行评测，需远端机器支持Docker。

```{tip}
- 数据集默认的**沙箱环境配置**可以在数据集配置中查看，例如[HumanEval](../benchmarks/humaneval.md)。
- **沙箱工作进程数** 需要根据本地机器的资源情况进行调整，建议设置为不超过本地CPU核心数的一半，例如本地有8核CPU，则建议设置为4，尤其是多语言镜像(`volcengine/sandbox-fusion`)，其消耗资源更多，建议适当减少工作进程数。
- 同一套沙箱基础设施同时驱动 [Agent 评测](agent/index.md) 中的 `environment='docker'`，可在 `agent_config.environment_extra` 中复用 `image` / `timeout` 等参数。
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