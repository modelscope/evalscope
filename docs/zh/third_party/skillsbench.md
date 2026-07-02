# SkillsBench

SkillsBench 评测 agent 是否能利用任务内置的 Agent Skills。每个任务包含 `task.md`、`environment/Dockerfile`、`environment/skills/`、`oracle/` 和 `verifier/`。EvalScope 会根据任务的 Dockerfile 本地构建镜像，在容器中运行 agent 或 oracle，然后执行任务自带的 `verifier/test.sh`，读取 `/logs/verifier/reward.txt` 作为分数。

## 前置条件

- 本地已安装并启动 Docker。
- 已 clone SkillsBench 仓库，并传入 `tasks/` 目录路径。
- 运行真实 agent 时需要配置 `agent_config`，例如 Codex external runner。
- verifier 脚本可能通过 apt/pip/uv 安装依赖，因此需要可用网络。

## Skill Mode

默认 `skill_mode` 是 `no-skill`。

- `no-skill`：构建临时上下文时删除 `environment/skills`，并清理 Dockerfile 中的 skill copy/path。
- `with-skill`：将任务内 `environment/skills` 注入镜像内 `/skills`，runner 启动前再安装到自己的 skill discovery path。

当前版本不支持 `self-gen`，也不会自动计算 lift。请分别运行 `no-skill` 和 `with-skill`，再使用 EvalScope 的 run comparison 能力比较两次结果。

## Oracle Smoke

先用 oracle 验证镜像、路径和 verifier contract：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='dummy',
    datasets=['skillsbench'],
    limit=1,
    dataset_args={
        'skillsbench': {
            'extra_params': {
                'tasks_dir': '/path/to/skillsbench/tasks',
                'task_ids': ['offer-letter-generator'],
                'runner': 'oracle',
                'skill_mode': 'no-skill',
            }
        }
    },
))
```

## 真实 Agent

`no-skill`：

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='your-model',
    datasets=['skillsbench'],
    limit=1,
    agent_config={
        'mode': 'external',
        'framework': 'codex',
        'environment': 'docker',
        'timeout': 900,
        'kwargs': {
            'auto_install': False,
        },
    },
    dataset_args={
        'skillsbench': {
            'extra_params': {
                'tasks_dir': '/path/to/skillsbench/tasks',
                'task_ids': ['offer-letter-generator'],
                'skill_mode': 'no-skill',
            }
        }
    },
))
```

`with-skill` 只需要改 `skill_mode`：

```python
dataset_args={
    'skillsbench': {
        'extra_params': {
            'tasks_dir': '/path/to/skillsbench/tasks',
            'task_ids': ['offer-letter-generator'],
            'skill_mode': 'with-skill',
        }
    }
}
```

## 镜像缓存

EvalScope 会为每个 task 和 skill mode 生成临时 build context，并基于 context hash 构建本地 Docker image。`no-skill` 和 `with-skill` 使用不同 cache key。首次运行会 build，后续相同上下文会复用本地镜像。设置 `force_rebuild=true` 可强制重建。

## 当前限制

- 默认只面向 SkillsBench 官方 `tasks/`，不自动包含 `tasks-extra/`。
- 当前 verifier contract 是 `verifier/test.sh` 写 `/logs/verifier/reward.txt`。
- 不支持完整 BenchFlow verifier 系统中的 `reward-kit`、`llm-judge`、`agent-judge`、`ors-episode` 或多 service verifier。
- 默认只保存 agent/verifier 关键日志和 metadata，不打包整个容器文件系统。
