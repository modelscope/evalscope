# Terminal-Bench 2.0

## 简介

Terminal-Bench v2 是一个命令行基准测试套件，用于评估 AI 代理在 89 个真实世界的多步骤终端任务上的表现。这些任务涵盖了从编译和调试到系统管理的广泛范围，所有任务都在隔离的容器中执行。每个任务都经过严格验证并自动评分（0/1），旨在推动前沿模型证明它们不仅能回答问题，还能采取行动。

EvalScope 集成了 Terminal-Bench v2 的评测功能，用户可以方便地使用 EvalScope 框架对支持 API 服务的模型进行评测。

- 项目地址：[harbor](https://github.com/laude-institute/harbor)
- 数据集地址：[terminal-bench-2](https://github.com/laude-institute/terminal-bench-2)

## 环境要求

```{important}
- **Python 版本**：必须使用 **Python >= 3.12**。
- **Docker 环境**：默认使用 Docker 运行评测，请确保本地已安装并启动 Docker Engine 和 Docker Compose。
- **网络连接**：
  - 数据集将直接从 GitHub 下载，请确保网络通畅。
  - 容器构建和运行时可能需要联网下载依赖，需保证环境网络稳定。
- **模型要求**：仅支持通过 API 服务评测被测模型（建议用 vLLM 等框架将本地模型先以服务形式暴露）。
```

## 安装依赖

```bash
pip install evalscope
# 安装 harbor 库 (Terminal-Bench 的核心依赖)
pip install harbor==0.1.28
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

    # 数据集配置
    datasets=['terminal_bench_v2'],
    dataset_args={
        'terminal_bench_v2': {
            'extra_params': {
                # 环境类型，默认为 'docker'
                # 可选: 'docker', 'daytona', 'e2b', 'modal'
                'environment_type': 'docker',
                
                # 代理类型，默认为 'terminus-2'
                # 可选: 'oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'
                # 'oracle'为标准答案代理, 可用于测试环境是否配置正确
                'agent_name': 'terminus-2',
                
                # 超时倍率，如果遇到超时错误可适当调大
                'timeout_multiplier': 1.0,

                # 最大交互轮数, 如果任务未完成可适当调大
                'max_turns': 200,
            }
        }
    },
    
    # 评测并发数 (建议根据 Docker 资源调整，每个并发会启动一个容器)
    eval_batch_size=1,

    # 限制评测样本数 (调试时可设置较小值, 正式评测去掉该参数)
    limit=10,  
)

# 开始评测
run_task(task_cfg)
```

## 参数说明

在 `dataset_args` 的 `extra_params` 中支持以下参数：

- `environment_type` (str): 运行基准测试的环境类型。默认为 `docker`。支持 `docker`, `daytona`, `e2b`, `modal`。
- `agent_name` (str): Harbor 中使用的代理类型。默认为 `terminus-2`。
- `timeout_multiplier` (float): 超时倍率。默认为 1.0。
- `max_turns` (int): 代理完成任务的最大交互轮数。默认为 200。

## 结果示例

评测需要较长的时间，请耐心等待。模型输出整体目录结构如下，其中模型推理的trials会自动保存在`outputs/<timestamp>/trials`目录。

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
+------------------+-------------------+----------+----------+-------+---------+---------+
| Model            | Dataset           | Metric   | Subset   |   Num |   Score | Cat.0   |
+==================+===================+==========+==========+=======+=========+=========+
| qwen3-coder-plus | terminal_bench_v2 | mean_acc | test     |     5 |     0.2 | default |
+------------------+-------------------+----------+----------+-------+---------+---------+
```

## 故障排除

1. **Docker 连接失败**：请检查 Docker Desktop 或 Docker Engine 是否正在运行，且当前用户有权限访问 Docker socket。
2. **数据集下载失败**：数据集通过 GitHub 下载，请检查网络连接或配置代理。
3. **Python 版本错误**：如果遇到语法错误或包兼容性问题，请确认使用的是 Python 3.12+。


