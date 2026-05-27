# Terminal-Bench-2.1


## 概述

Terminal-Bench v2.1 是 Terminal-Bench 2.0 的改进版本，修复了 26 项任务中的 bug、调整了超时设置，并增强了对奖励作弊（reward hacking）的防范。建议在新评估中优先使用 v2.1 而非 v2.0。

## 任务描述

- **任务类型**：命令行智能体评估
- **输入**：终端任务描述
- **输出**：通过智能体操作完成任务
- **领域**：系统管理、编译、调试、文件操作

## 主要特性

- 在 Terminal-Bench 2.0 基础上验证并修复了 26 项任务
- 提升了对奖励作弊的鲁棒性
- 使用隔离的容器执行环境
- 采用二元评分（0/1）并支持自动验证
- 支持多种智能体类型（terminus-2、claude-code、codex 等）

## 评估说明

- 需要 **Python>=3.12** 并运行 `pip install evalscope[terminal_bench]`
- 支持的环境选项：docker、daytona、e2b、modal
- 可配置智能体类型和超时设置
- 最大交互轮次可配置（默认：200）
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/terminal_bench.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `terminal_bench_v2_1` |
| **数据集ID** | [latest](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 89 |
| 提示词长度（平均） | 0 字符 |
| 提示词长度（最小/最大） | 0 / 0 字符 |

## 样例示例

**子集**: `test`

```json
{
  "input": [
    {
      "id": "d3a63987",
      "content": ""
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "path": null,
    "git_url": null,
    "git_commit_id": null,
    "name": "terminal-bench/write-compressor",
    "ref": "sha256:d9ddd9a8e925e2c566b37b2492cbf995afecefe58874e4043ef78d7f3c892c7e",
    "overwrite": false,
    "download_dir": "/Users/yunlin/.cache/evalscope/terminal_bench_v2_1",
    "source": "terminal-bench/terminal-bench-2-1"
  }
}
```

## 提示模板

**提示模板:**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `environment_type` | `str` | `docker` | 运行基准测试的环境类型。可选值：['docker', 'daytona', 'e2b', 'modal'] |
| `agent_name` | `str` | `terminus-2` | Harbor 中使用的智能体类型。仅 terminus-2 使用 evalscope 模型进行推理；其他智能体（如 claude-code、codex 等）作为独立 CLI 工具运行，需自行提供 API 密钥。可选值：['oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'] |
| `timeout_multiplier` | `float` | `1.0` | 超时倍率。若出现超时错误，可考虑增大该值。 |
| `max_turns` | `int` | `200` | 智能体完成任务的最大交互轮次。 |
| `environment_kwargs` | `dict` | `{}` | 传递给 Harbor EnvironmentConfig 的额外参数。支持的键包括：override_cpus、override_memory_mb、override_storage_mb、override_gpus、force_build、delete、env 等。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v2_1 \
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
    datasets=['terminal_bench_v2_1'],
    dataset_args={
        'terminal_bench_v2_1': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```