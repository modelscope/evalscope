# Terminal-Bench-2.0

## 概述

Terminal-Bench v2 是一套命令行基准测试套件，用于评估 AI 代理在 89 个真实世界的多步骤终端任务中的表现。这些任务涵盖编译、调试到系统管理等多个方面，并在隔离的容器环境中运行，配有严格的验证机制。

## 任务描述

- **任务类型**：命令行代理评估  
- **输入**：终端任务说明  
- **输出**：通过代理操作完成任务  
- **领域**：系统管理、编译、调试、文件操作  

## 主要特性

- 包含 89 个真实世界的终端任务  
- 要求完成多步骤任务  
- 在隔离的容器环境中执行  
- 采用二值评分（0/1）并支持自动验证  
- 支持多种代理类型（如 terminus-2、claude-code、codex 等）

## 评估说明

- 需要 **Python>=3.12** 并安装 `pip install harbor==0.1.28`  
- 支持的环境选项：docker、daytona、e2b、modal  
- 可配置代理类型和超时设置  
- 最大交互轮数可配置（默认：200）  
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/terminal_bench.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `terminal_bench_v2` |
| **数据集 ID** | [terminal-bench-2.git](https://github.com/laude-institute/terminal-bench-2.git) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `environment_type` | `str` | `docker` | 运行基准测试的环境类型。可选值：['docker', 'daytona', 'e2b', 'modal'] |
| `agent_name` | `str` | `terminus-2` | Harbor 中使用的代理类型。可选值：['oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'] |
| `timeout_multiplier` | `float` | `1.0` | 超时倍数。如果出现超时错误，可考虑增大该值。 |
| `max_turns` | `int` | `200` | 代理完成任务的最大交互轮数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v2 \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['terminal_bench_v2'],
    dataset_args={
        'terminal_bench_v2': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```