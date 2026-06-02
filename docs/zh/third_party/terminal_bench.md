# Terminal-Bench

## 简介

Terminal-Bench 是一个命令行基准测试套件，用于评估 AI 代理在真实世界的多步骤终端任务上的表现。这些任务涵盖了从编译和调试到系统管理的广泛范围，所有任务都在隔离的容器中执行。每个任务都经过严格验证并自动评分（0/1），旨在推动前沿模型证明它们不仅能回答问题，还能采取行动。

EvalScope 支持两个版本：

- **`terminal_bench_v2`**：原始 89 个任务的基准测试（Terminal-Bench 2.0）。
- **`terminal_bench_v2_1`**：改进版本，修复了 26 个任务的 bug、超时问题和防奖励作弊机制（Terminal-Bench 2.1，推荐使用）。

相关链接：

- 项目地址：[harbor](https://github.com/laude-institute/harbor)
- 数据集（Hub）：[terminal-bench-2](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2/latest) | [terminal-bench-2-1](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest)

## 环境要求

```{important}
- **Python 版本**：必须使用 **Python >= 3.12**。
- **Docker 环境**：默认使用 Docker 运行评测，请确保运行 EvalScope 的环境中已安装并启动 Docker Engine、Docker Compose 以及 `docker` 命令行工具。
- **网络连接**：
  - 数据集从 Harbor Hub 下载，请确保网络通畅。
  - 容器构建和运行时可能需要联网下载依赖，需保证环境网络稳定。
- **模型要求**：仅 `terminus-2` 代理会使用你配置的模型进行推理。其他代理（claude-code、codex 等）是独立的 CLI 工具，使用各自的 API key。
```

## 安装依赖

```bash
pip install evalscope[terminal_bench]
```

## 使用方法

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
                # 使用各自的 API key，忽略此处配置的模型。
                'agent_name': 'terminus-2',

                # 超时倍率，如果遇到超时错误可适当调大
                'timeout_multiplier': 1.0,

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
- `max_turns` (int)：代理完成任务的最大交互轮数。默认为 200。
- `environment_kwargs` (dict)：传递给 Harbor `EnvironmentConfig` 的额外参数，用于配置容器资源限制等。支持的 key 包括：`override_cpus`、`override_memory_mb`、`override_storage_mb`、`override_gpus`、`force_build`、`delete`、`env` 等。

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
| qwen3-coder-plus | terminal_bench_v2_1 | mean_acc | test     |     5 |     0.2 | default |
+------------------+---------------------+----------+----------+-------+---------+---------+
```

## 故障排除

1. **Docker 连接失败**：请检查 Docker Desktop 或 Docker Engine 是否正在运行，且当前用户有权限访问 Docker socket。
2. **Docker CLI 未找到**：如果 EvalScope 在容器内运行，仅挂载 `/var/run/docker.sock` 是不够的，容器内还需要安装 `docker` 命令行工具。
3. **数据集下载失败**：数据集通过 Harbor Hub 下载，请检查网络连接或配置代理。
4. **Python 版本错误**：如果遇到语法错误或包兼容性问题，请确认使用的是 Python 3.12+。
