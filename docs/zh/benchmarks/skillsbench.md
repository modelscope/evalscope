# SkillsBench


## 概述

SkillsBench 用于评估编码智能体是否能够发现并应用任务捆绑的 Agent 技能。每个任务包含一条指令、一个可选的技能目录、一个 Docker 环境、一个参考解决方案（oracle solution）和一个验证器（verifier）。EvalScope 会构建任务的 Docker 镜像，在该镜像中运行所选的智能体或参考方案，然后执行任务验证器。

## 任务描述

- **任务类型**：智能体技能使用 / 工具辅助任务完成
- **输入**：来自 `task.md` 的自然语言任务提示
- **输出**：在任务容器内生成的文件或状态变更，由任务验证器评分
- **数据集**：通过 `extra_params.tasks_dir` 提供的本地 SkillsBench 任务仓库
- **环境**：基于 `environment/Dockerfile` 构建的每个任务专属 Docker 镜像
- **技能**：可选的任务捆绑技能，位于 `environment/skills`
- **指标**：来自 `/logs/verifier/reward.txt` 的 `score`；当 `score > 0` 时，`success` 为 1

## 核心特性

- 为每个选定任务构建或复用基于内容哈希的 Docker 镜像。
- 在单次 EvalScope 运行中，以 `no-skill` 或 `with-skill` 模式运行任务，不会混合两种模式。
- 通过 EvalScope 的智能体技能运行时注入任务捆绑技能，而非将运行器特定的技能路径硬编码进镜像。
- 支持 EvalScope 原生智能体和通过共享智能体环境接口的外部智能体运行器。
- 将验证器的标准输出、奖励值以及可选的 CTRF 工件保存至运行输出目录。

## 评估说明

- 默认 `skill_mode` 为 `no-skill`。
- 分别运行 `no-skill` 和 `with-skill` 模式，然后使用 EvalScope 的运行对比工具进行比较。
- 此适配器版本不支持 `self-gen` 和 `tasks-extra`。
- `runner='oracle'` 会运行官方的 `oracle/solve.sh` 用于冒烟测试；智能体运行需提供 `agent_config`。
- 验证器执行 `verifier/test.sh`，可能会从网络安装依赖项。

## 评分与对比

- `score` 是从 `/logs/verifier/reward.txt` 解析出的验证器奖励值。
- `success` 在本地计算：当 `score > 0` 时为 `1.0`，否则为 `0.0`。
- EvalScope 不会自动计算 `no-skill` 与 `with-skill` 的差异；需分别运行两种模式并对比其报告。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `skillsbench` |
| **数据集ID** | `skillsbench` |
| **论文** | N/A |
| **标签** | `Agent`, `MultiTurn` |
| **指标** | `score` |
| **默认示例数** | 0-shot |
| **评估划分** | `N/A` |


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
| `tasks_dir` | `str` | `` | SkillsBench 任务目录的路径。 |
| `task_ids` | `list` | `[]` | 可选的任务 ID 列表，用于指定运行哪些任务。 |
| `skill_mode` | `str` | `no-skill` | SkillsBench 技能模式。选项：['no-skill', 'with-skill'] |
| `force_rebuild` | `bool` | `False` | 强制重新构建任务 Docker 镜像。 |
| `agent_timeout_sec` | `float` | `None` | 覆盖任务智能体的超时时间。 |
| `verifier_timeout_sec` | `float` | `None` | 覆盖任务验证器的超时时间。 |
| `runner` | `str` | `agent` | 使用 "oracle" 来运行 oracle/solve.sh 而非智能体。选项：['agent', 'oracle'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets skillsbench \
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
    datasets=['skillsbench'],
    dataset_args={
        'skillsbench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
