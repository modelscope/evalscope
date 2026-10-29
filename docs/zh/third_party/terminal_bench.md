# Terminal-Bench

## 简介

Terminal-Bench 是一个命令行基准测试套件，用于评估 AI 代理在真实世界的多步骤终端任务上的表现。这些任务涵盖了从编译和调试到系统管理的广泛范围，所有任务都在隔离的容器中执行。每个任务都经过严格验证并自动评分（0/1），旨在推动前沿模型证明它们不仅能回答问题，还能采取行动。

EvalScope 支持三个版本：

- **`terminal_bench_v2`**：原始 89 个任务的基准测试（Terminal-Bench 2.0）。
- **`terminal_bench_v2_1`**：改进版本，修复了 26 个任务的 bug、超时问题和防奖励作弊机制（Terminal-Bench 2.1，推荐使用）。
- **`terminal_bench_v4`**：Terminal-Bench 4.0，固定上游版本 **4.0.0**，包含 66 个任务，全部使用独立 verifier 环境。结果不能与 2.0／2.1 直接比较。

相关链接：

- 项目地址：[harbor](https://github.com/harbor-framework/harbor)
- 数据集（Hub）：[terminal-bench-2](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2/latest) | [terminal-bench-2-1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest) | [terminal-bench 4.0](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4)

## 环境要求

```{important}
- **Python 版本**：必须使用 **Python >= 3.12**。
- **Harbor 版本**：要求 **Harbor >= 0.14.0, < 1.0.0**。旧版本无法处理 4.0 的 sidecar artifact 和 verifier 收集钩子。
- **Docker 环境**：默认使用 Docker 运行评测，请确保运行 EvalScope 的环境中已安装并启动 Docker Engine、Docker Compose 以及 `docker` 命令行工具。
- **网络连接**：
  - 数据集从 Harbor Hub 下载，请确保网络通畅。
  - 容器构建和运行时可能需要联网下载依赖，需保证环境网络稳定。
- **模型要求**：仅 `terminus-2` 代理会使用你配置的模型进行推理。其他代理（claude-code、codex 等）是独立的 CLI 工具，使用各自的 API key。
```

## 安装依赖

```bash
pip install --upgrade 'evalscope[terminal_bench]'
```

## 使用方法

### Terminal-Bench 4.0

默认加载 **4.0.0** 的全部 **66 个任务**，使用 Docker 和 `terminus-2`，继承官方任务的资源和时限（agent 为 8 小时），并采用 Harbor 默认回合上限（`max_turns=None`）。所有任务都有独立 verifier，其中 11 个包含 Docker Compose 服务，3 个要求 H100 GPU。全量运行需要支持 H100 的 Linux Docker 主机、足够的多容器资源，以及依赖下载所需的网络连接。

本地 CPU 调试应显式选取任务；`limit` 只限制数量，不能保证排除 GPU 或多容器任务：

```python
import os
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='qwen3-coder-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['terminal_bench_v4'],
    dataset_args={
        'terminal_bench_v4': {
            'extra_params': {
                'task_names': ['terminal-bench/bun-sourcemap-leak'],
            },
        },
    },
    eval_batch_size=1,
))
```

`task_names` 支持 Harbor 任务全名或 glob，例如 `terminal-bench/ctr-*`。筛选在 EvalScope 的 shuffle、limit、repeats 和索引之前执行。去掉 `task_names` 即加载全部 66 个任务。在 `TaskConfig` 设置 `repeats=5` 可让每个任务独立运行五次，报告聚合有效 reward 的平均值。子集结果仅用于调试，不代表完整基准分数。

使用 Modal 时，安装额外依赖并配置账号凭据：

```bash
pip install --upgrade 'harbor[modal]>=0.14.0,<1.0.0'
modal token new
```

然后在 `extra_params` 中设置 `'environment_type': 'modal'`。Harbor 根据任务配置选择 GPU，并管理独立 verifier 和 Compose 服务；账号需要能够申请这些资源。

可通过 `agent_name` 选择外部 CLI agent，例如 `'claude-code'` 或 `'codex'`。这些工具自行处理推理和认证，使用各自凭据；只有 `terminus-2` 使用 EvalScope 的模型 API 桥接。
使用 `eval_type='mock_llm'` 可避免创建不会用到的 EvalScope API 客户端；将 `model` 设置为所选 CLI 支持的模型，并在启动 EvalScope 前配置该工具的凭据。模型名称仍会传给 Harbor，但 `api_url` 和 `api_key` 不会配置外部 CLI。

### Terminal-Bench 2.0 / 2.1

以下示例展示如何使用 evalscope 评测 `qwen3-coder-plus` 模型。

```python
import os
from evalscope import TaskConfig, run_task

# 配置任务
task_cfg = TaskConfig(
    # 模型配置
    model='qwen3-coder-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用 OpenAI 兼容的服务评测

    # 数据集配置（使用 'terminal_bench_v2_1' 表示 v2.1，或 'terminal_bench_v2' 表示 v2.0）
    datasets=['terminal_bench_v2_1'],
    dataset_args={
        'terminal_bench_v2_1': {
            'extra_params': {
                # 环境类型，默认为 'docker'
                # 可选：'docker', 'daytona', 'e2b', 'modal'
                'environment_type': 'docker',

                # 代理类型，默认为 'terminus-2'
                # 仅 'terminus-2' 使用你配置的模型进行推理。
                # 其他代理（claude-code、codex 等）是独立 CLI 工具，
                # 使用各自的 API key；模型名称会传给 Harbor。
                'agent_name': 'terminus-2',

                # 超时倍率，如果遇到超时错误可适当调大
                'timeout_multiplier': 1.0,

                # 复现实验的绝对超时。不要与同一阶段的倍率同时设置。
                'agent_timeout_sec': 3 * 60 * 60,
                'verifier_timeout_sec': 3 * 60 * 60,

                # Qwen 3.6 公开协议的容器资源。
                'environment_kwargs': {
                    'override_cpus': 32,
                    'override_memory_mb': 48 * 1024,
                },

                # 最大交互轮数，如果任务未完成可适当调大
                'max_turns': 200,
            }
        }
    },

    # 评测并发数（建议根据 Docker 资源调整，每个并发会启动一个容器）
    eval_batch_size=1,

    # 限制评测样本数（调试时可设置较小值，正式评测去掉该参数）
    limit=10,
)

# 开始评测
run_task(task_cfg)
```

## 参数说明

在 `dataset_args` 的 `extra_params` 中支持以下参数：

- `environment_type` (str)：运行基准测试的环境类型。默认为 `docker`。支持 `docker`、`daytona`、`e2b`、`modal`。
- `agent_name` (str)：Harbor 中使用的代理类型。默认为 `terminus-2`。仅 `terminus-2` 使用 evalscope 配置的模型进行推理；其他代理（claude-code、codex、opencode 等）是独立 CLI 工具，使用各自的 API key。
- `timeout_multiplier` (float)：超时倍率。默认为 1.0。
- `agent_timeout_sec` / `verifier_timeout_sec` (float)：阶段最终超时秒数，不会再被 `timeout_multiplier` 二次放大。
- `agent_timeout_multiplier` / `verifier_timeout_multiplier` (float)：覆盖全局倍率的阶段倍率；不得与同阶段绝对超时同时设置。
- `max_turns` (int 或 None)：最大交互轮数。2.0／2.1 默认为 200，4.0 默认为 None（采用 Harbor 默认值）。
- `task_names` (list，仅 4.0)：可选的 Harbor 任务全名或 glob，默认为 None，即加载全部任务。
- `environment_kwargs` (dict)：传递给 Harbor `EnvironmentConfig` 的额外参数，用于配置容器资源限制等。支持的 key 包括：`override_cpus`、`override_memory_mb`、`override_storage_mb`、`override_gpus`、`force_build`、`delete`、`env` 等。

`override_storage_mb` 是否生效取决于 Docker storage driver 和宿主文件系统是否支持容器配额。容器构建和 verifier
脚本可能需要从网络安装依赖；请通过 `environment_kwargs.env` 注入已批准的镜像或代理。EvalScope 不会根据 verifier
输出推断基础设施故障：verifier 正常写出的 reward `0` 仍按模型失败计分。

## 结果示例

评测需要较长的时间，请耐心等待。模型输出整体目录结构如下，其中模型推理的 trials 会自动保存在 `outputs/<timestamp>/trials` 目录。

```text
.
├── configs
│   └── task_config_07906d.yaml
├── logs
│   └── eval_log.log
├── predictions
│   └── qwen-plus
└── trials
    ├── adaptive-rejection-sampler__44jsCkg
    ├── bn-fit-modify__BDZ7XN8
    ├── break-filter-js-from-html__xSUatRJ
    ├── build-cython-ext__Cj6Mi7X
    ├── build-pmars__ghgk7h8
    └── build-pov-ray__JWy4tcZ
```

评测完成后会输出类似如下的结果表格：

```text
+------------------+---------------------+----------+----------+-------+---------+---------+
| Model            | Dataset             | Metric   | Subset   |   Num |   Score | Cat.0   |
+==================+=====================+==========+==========+=======+=========+=========+
| qwen3-coder-plus | terminal_bench_v2_1 | Accuracy ↑ | test     |     5 |     20% | default |
+------------------+---------------------+----------+----------+-------+---------+---------+
```

## 故障排除

1. **Docker 连接失败**：请检查 Docker Desktop 或 Docker Engine 是否正在运行，且当前用户有权限访问 Docker socket。
2. **Docker CLI 未找到**：如果 EvalScope 在容器内运行，仅挂载 `/var/run/docker.sock` 是不够的，容器内还需要安装 `docker` 命令行工具。
3. **数据集下载失败**：数据集通过 Harbor Hub 下载，请检查网络连接或配置代理。
4. **Python 版本错误**：如果遇到语法错误或包兼容性问题，请确认使用的是 Python 3.12+。
