# Terminal-Bench-4.0


## 概述

Terminal-Bench 4.0 使用官方 Harbor harness 对智能体在 66 个具有挑战性的终端任务上进行评估。该版本校准了任务资源，修复了任务指令和验证器，并移除了不再能有效区分智能体性能的任务。

## 任务描述

- **任务类型**：命令行智能体评估
- **输入**：任务指令和一个隔离的运行时环境
- **输出**：智能体生成的产物及任务完成情况，由官方验证器检查
- **领域**：软件工程、机器学习、安全、科学和运维

## 主要特性

- 固定上游数据集版本为 **4.0.0**，采用基于内容寻址的任务版本
- 包含 66 个任务，其中 11 个为多容器任务，3 个任务需要 H100 GPU
- 所有任务均配备独立的验证器环境，包括一个使用 GPU 的验证器任务
- 官方设定每个任务的智能体超时时间为 8 小时；各任务具有特定的 CPU、内存和验证器资源配额
- 支持可配置的 Harbor 智能体、显式任务名称过滤以及重复评估

## 评估说明

- 需要 **Python>=3.12**、**Harbor>=0.14.0,<1.0.0**，并执行 `pip install 'evalscope[terminal_bench]'`
- 得分是有效官方验证器奖励（0/1）的平均值；失败的试验和无效奖励不计入有效得分
- 默认使用 Docker 和 terminus-2；terminus-2 使用配置的 EvalScope 模型，而外部 CLI 智能体则使用其自身的 API 凭据
- 完整评估需要具备 H100 GPU 访问权限和多容器支持的 Docker 主机；也可选择使用 Modal，需安装 `harbor[modal]` 并提供账户凭据
- 任务资源和超时设置继承自上游；max_turns 默认为 None，使用 Harbor 智能体的默认值
- 可通过 task_names 指定用于调试的子集，通过 repeats 设置重复试验次数；子集结果及旧版基准测试的结果不能直接与完整的 4.0 评估结果进行比较
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/terminal_bench.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `terminal_bench_v4` |
| **数据集ID** | [4](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 66 |
| 提示词长度（平均） | 0 字符 |
| 提示词长度（最小/最大） | 0 / 0 字符 |

## 样例示例

**子集**: `test`

```json
{
  "input": [
    {
      "id": "5706a469",
      "content": ""
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "path": null,
    "git_url": null,
    "git_commit_id": null,
    "name": "terminal-bench/layout-config-recreation2",
    "ref": "sha256:193312b14b1dcb6ef551efff691cbb708b2032f58879b0b19efe88a2cddc9deb",
    "overwrite": false,
    "download_dir": "~/.cache/evalscope/terminal_bench_v4",
    "source": "terminal-bench/terminal-bench"
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
| `environment_type` | `str` | `docker` | 运行基准测试的环境类型。选项：['docker', 'daytona', 'e2b', 'modal'] |
| `agent_name` | `str` | `terminus-2` | Harbor 中使用的智能体类型。仅 terminus-2 使用 evalscope 模型进行推理；其他智能体（如 claude-code、codex 等）作为独立 CLI 工具运行，需自行提供 API 密钥。选项：['oracle', 'terminus-2', 'claude-code', 'codex', 'qwen-coder', 'openhands', 'opencode', 'mini-swe-agent'] |
| `timeout_multiplier` | `float` | `1.0` | 超时倍率。若出现超时错误，可考虑增大此值。 |
| `agent_timeout_sec` | `float` | `None` | 智能体阶段最终超时时间（秒）。不能与 agent_timeout_multiplier 同时使用。 |
| `verifier_timeout_sec` | `float` | `None` | 验证器阶段最终超时时间（秒）。不能与 verifier_timeout_multiplier 同时使用。 |
| `agent_timeout_multiplier` | `float` | `None` | 智能体阶段超时倍率。覆盖 timeout_multiplier 在智能体阶段的设置。 |
| `verifier_timeout_multiplier` | `float` | `None` | 验证器阶段超时倍率。覆盖 timeout_multiplier 在验证器阶段的设置。 |
| `max_turns` | `int` | `None` | 智能体最大交互轮次。None 表示使用 Harbor 智能体的默认值。 |
| `environment_kwargs` | `dict` | `{}` | 传递给 Harbor EnvironmentConfig 的额外参数。支持的键包括：override_cpus、override_memory_mb、override_storage_mb、override_gpus、force_build、delete、env 等。 |
| `task_names` | `list` | `None` | 可选的 Harbor 任务名称或通配符模式，例如 terminal-bench/ctr-optimization。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets terminal_bench_v4 \
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
    datasets=['terminal_bench_v4'],
    dataset_args={
        'terminal_bench_v4': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
