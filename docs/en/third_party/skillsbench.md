# SkillsBench

SkillsBench evaluates whether agents can use task-bundled Agent Skills. Each task contains `task.md`, `environment/Dockerfile`, `environment/skills/`, `oracle/`, and `verifier/`. EvalScope builds a local Docker image from the task environment, runs the agent or oracle in a container, then executes the task's `verifier/test.sh` and reads `/logs/verifier/reward.txt` as the score.

## Prerequisites

- Docker is installed and running.
- The SkillsBench repository is cloned locally, and you pass the `tasks/` directory path.
- Real agent runs require `agent_config`, for example the Codex external runner.
- External runners execute inside the task container. Codex/OpenCode/Gemini CLI runners probe for the CLI first and can install it during runner setup when `auto_install=True`.
- Verifier scripts may install dependencies through apt/pip/uv, so network access is usually required.

## Skill Mode

The default `skill_mode` is `no-skill`.

- `no-skill`: EvalScope removes `environment/skills` from the temporary build context and strips skill copy/path residues from the Dockerfile.
- `with-skill`: EvalScope injects task-bundled skills into `/skills`; the runner installs them into its own skill discovery path before launch.

`self-gen` is not supported in this version. EvalScope does not automatically compute lift. Run `no-skill` and `with-skill` separately, then compare the two runs with EvalScope's run comparison tools.

## Oracle Smoke

Use oracle first to validate the image, paths, and verifier contract:

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

## Real Agent

`no-skill`:

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
            'auto_install': True,
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

For `with-skill`, only change `skill_mode`:

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

## Image Cache

EvalScope creates a temporary build context for each task and skill mode, then tags the local Docker image from the context hash. `no-skill` and `with-skill` use different cache keys. The first run builds the image; later runs with the same context reuse it. Set `force_rebuild=true` to rebuild.

The task image remains the source of truth for task dependencies and verifier inputs. If that image already contains the external CLI, set the runner's `auto_install` to `False`; otherwise leave `auto_install=True` so the runner installs the CLI inside the task container during setup.

## Current Limits

- The adapter targets the official SkillsBench `tasks/` set by default and does not include `tasks-extra/` automatically.
- The supported verifier contract is `verifier/test.sh` writing `/logs/verifier/reward.txt`.
- Full BenchFlow verifier features such as `reward-kit`, `llm-judge`, `agent-judge`, `ors-episode`, and multi-service verifiers are not supported.
- EvalScope saves key agent/verifier logs and metadata by default, not the full container filesystem.
