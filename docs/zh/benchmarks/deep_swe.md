# DeepSWE


## 概述

DeepSWE 是一个用于评估仓库级软件工程任务的编码智能体基准测试。EvalScope 通过 Pier 集成该基准，并将每个基准样本作为一项 Pier Python API 任务运行。

## 任务描述

- **任务类型**：智能体软件工程
- **输入**：包含任务元数据和验证器资源的 DeepSWE 任务目录
- **输出**：由 Pier 内置智能体生成的代码仓库补丁
- **评分方式**：二值验证器奖励，以 `acc` 形式暴露

## 评估说明

- 要求 **Python>=3.12**、Docker，以及执行 `pip install evalscope[deep_swe]`
- 数据集默认使用 ModelScope 上的 `evalscope/deep-swe`
- DeepSWE 在 EvalScope 中通过 Pier 的 Docker 环境运行
- 对于不支持 Responses API 的 OpenAI 兼容提供商，请使用 `pier_agent_kwargs={'model_class': 'litellm'}`

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `deep_swe` |
| **数据集ID** | [evalscope/deep-swe](https://modelscope.cn/datasets/evalscope/deep-swe/summary) |
| **论文** | 无 |
| **标签** | `Agent`, `Coding`, `MultiTurn` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 113 |
| 提示词长度（平均） | 2158.07 字符 |
| 提示词长度（最小/最大） | 471 / 5385 字符 |

## 样例示例

**子集**: `test`

```json
{
  "input": [
    {
      "id": "f61040e0",
      "content": "Add a new `errorStack` constructor option to SuperJSON. Omitting it leaves existing Error behavior unchanged.\n\nThe option shape is `{ mode?, normalizeNewlines?, trimLeadingWhitespace?, maxStackLines?, stripInternalFrames?, redactPaths?, inclu ... [TRUNCATED 3577 chars] ... ): Processor | undefined`. `normalizeErrorStackOptions` returns `undefined` for any non-object input (`null`, `undefined`, strings).\n\nBefore writing, read through the existing error serialization logic and the `allowedErrorProps` mechanism.\n\n"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "ext_id": "kh701jywhzgddknqwzsq6npjv98226tq",
    "task_id": "superjson-error-stack-serialization",
    "display_title": "Add error stack serialization to SuperJSON",
    "display_description": "Add configurable serialization and restoration of error stacks, stack frames, causes, and sanitization in SuperJSON.",
    "repo": "flightcontrolhq/superjson",
    "repository_url": "https://github.com/flightcontrolhq/superjson.git",
    "original_title": "Error Stack Serialization Support",
    "category": "feature_request",
    "language": "typescript",
    "task_path": "~/.cache/evalscope/deep_swe/snapshots/evalscope/deep-swe/tasks/superjson-error-stack-serialization",
    "task_toml_path": "~/.cache/evalscope/deep_swe/snapshots/evalscope/deep-swe/tasks/superjson-error-stack-serialization/task.toml",
    "instruction": "Add a new `errorStack` constructor option to SuperJSON. Omitting it leaves existing Error behavior unchanged.\n\nThe option shape is `{ mode?, normalizeNewlines?, trimLeadingWhitespace?, maxStackLines?, stripInternalFrames?, redactPaths?, inclu ... [TRUNCATED 3577 chars] ... ): Processor | undefined`. `normalizeErrorStackOptions` returns `undefined` for any non-object input (`null`, `undefined`, strings).\n\nBefore writing, read through the existing error serialization logic and the `allowedErrorProps` mechanism.\n\n"
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
| `task_ids` | `list` | `[]` | 可选的 DeepSWE 任务 ID 列表，用于指定评估范围。 |
| `languages` | `list` | `[]` | 可选的任务语言过滤器，基于清单元数据。 |
| `categories` | `list` | `[]` | 可选的任务类别过滤器，基于清单元数据。 |
| `sample_seed` | `int` | `` | 可选的确定性打乱种子，在限制样本数量前应用。 |
| `pier_agent_kwargs` | `dict` | `{}` | 传递给 Pier AgentConfig.kwargs 的额外关键字参数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets deep_swe \
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
    datasets=['deep_swe'],
    dataset_args={
        'deep_swe': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```