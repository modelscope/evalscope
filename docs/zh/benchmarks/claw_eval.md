# Claw-Eval


## 概述

Claw-Eval 用于评估助手代理在真实个人助理工作流中的表现，这些工作流需要使用工具、访问文件和固定资源（fixtures）、处理多模态输入，并支持模拟用户交互。EvalScope 运行官方指定版本的 Claw-Eval Python 执行器、Docker 沙箱和评分器，同时将每个 Claw-Eval 任务作为标准的 EvalScope 样本进行处理，以支持缓存、重复执行、并行运行、报告生成以及仪表盘轨迹审查。

## 任务描述

- **任务类型**：涉及工具调用、沙箱文件、多模态固定资源及可选模拟用户轮次的代理型个人助理任务。
- **数据集**：ModelScope 上的 `claw-eval/Claw-Eval`。
- **子集**：`general`、`multimodal` 和 `multi_turn`；当前 ModelScope 清单包含 300 个任务（161 个 general、101 个 multimodal、38 个 multi_turn）。可通过 `subset_list` 选择子集。
- **输出**：官方 Claw-Eval 评分与 JSONL 轨迹、EvalScope 样本级评审结果、分组汇总指标，以及仪表盘渲染的代理轨迹。

## 评估说明

- 需要 Python 3.11+，并从指定源代码提交安装官方包：
  `pip install "claw-eval[sandbox,mock,web] @ git+https://github.com/claw-eval/claw-eval.git@d3f02d4938ab0832377d90535013def2b1a2fdc0"`。
- 安装的包提供 Claw-Eval 执行器 API。EvalScope 同时缓存相同版本的源代码归档，因为 `tasks/` 和 `Dockerfile.agent` 是运行时资产，随后从 ModelScope 加载任务清单和固定资源。
- 完整的固定资源从 ModelScope 下载（`data/fixtures.tar.gz`），并在执行前链接到官方任务目录中。该归档较大；建议在试运行时使用 `limit` 或 `extra_params.task_ids` 参数。
- 每个选定的 Claw-Eval 任务对应一个 EvalScope 样本。官方评分对每个样本仅运行一次；如需对同一任务进行多次试验，请使用 EvalScope 的 `repeats` 参数；如需任务级并发执行，请使用 `eval_batch_size`。
- Claw-Eval 使用官方 Docker 沙箱镜像运行。若本地缺少 `claw-eval-agent:latest` 镜像，EvalScope 会自动基于缓存的官方 `Dockerfile.agent` 构建。首次运行可能较慢。
- EvalScope 的 `use_cache` 功能可恢复已完成的任务级样本。Claw-Eval 轨迹 JSONL 文件存储在 `outputs/.../claw_eval/<split>/traces` 目录下，并转换为 EvalScope 代理轨迹以供仪表盘可视化。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `claw_eval` |
| **数据集ID** | [claw-eval/Claw-Eval](https://modelscope.cn/datasets/claw-eval/Claw-Eval/summary) |
| **论文** | 无 |
| **标签** | `Agent`, `MultiModal`, `MultiTurn` |
| **指标** | `avg_score`, `pass_at_k`, `pass_hat_k`, `error_rate` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 300 |
| 提示词长度（平均） | 47.36 字符 |
| 提示词长度（最小/最大） | 30 / 60 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `general` | 161 | 46.47 | 34 | 58 |
| `multimodal` | 101 | 49.75 | 30 | 60 |
| `multi_turn` | 38 | 44.79 | 35 | 51 |

## 样例示例

**子集**: `general`

```json
{
  "input": [
    {
      "id": "a34b7f6b",
      "content": "Run Claw-Eval task T001zh_email_triage."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "general",
  "metadata": {
    "task_id": "T001zh_email_triage",
    "split": "general",
    "task_name": "",
    "difficulty": "",
    "dataset_id": "claw-eval/Claw-Eval",
    "dataset_hub": "modelscope"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `task_ids` | `list` | `[]` | 可选，在子集过滤后精确指定要运行的 Claw-Eval 任务 ID 列表。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets claw_eval \
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
    datasets=['claw_eval'],
    dataset_args={
        'claw_eval': {
            # subset_list: ['general', 'multimodal', 'multi_turn']  # 可选，评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
