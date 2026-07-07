# PerspectiveGap Role Assignment

## 概述

PerspectiveGap 评估模型能否为多智能体系统编写编排提示，并只向每个子智能体路由其完成任务所需的上下文。

## 任务

- `perspective_gap_role_assignment`：为每个角色选择可见的 fragment ID，并返回 JSON 对象。
- `perspective_gap_prompt_writing`：为每个角色编写一个 markdown prompt section，并且只包含该角色需要的 fragments。

## 数据

该基准使用 ModelScope 数据集 `evalscope/PerspectiveGap`，其中包含已发布的 `test` split。也可以通过 `dataset_args.<task>.local_path` 传入字段相同的本地 JSONL 镜像。

## 评分

分数由官方 PerspectiveGap 仓库中的 `perspective_gap.scoring` 计算。scorer 采用懒加载，因此列出 EvalScope benchmarks 时不需要安装这个可选依赖。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `perspective_gap_role_assignment` |
| **数据集ID** | [evalscope/PerspectiveGap](https://modelscope.cn/datasets/evalscope/PerspectiveGap/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2606.08878) |
| **标签** | `Agent`, `InstructionFollowing` |
| **指标** | `strict_pass`, `net_match_score`, `required_coverage`, `boundary_precision`, `distractor_leakage` |
| **默认 Shot 数** | 0-shot |
| **评测划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 220 |
| 提示词长度（均值） | 11329.68 chars |
| 提示词长度（最小/最大） | 3714 / 19088 chars |

## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "ab2e63cc",
      "content": "I need you to set up a 2-agent pipeline for bug bounty.\nThe two roles are coder and reviewer: coder is responsible for finding bugs, reviewer is responsible for judging whether what the coder found counts as a bug.\nAfter each bug the coder fi ... [TRUNCATED 3528 chars] ... t's prompt should contain only the information that agent needs to do its job. For each sub-agent (coder, reviewer), which items should go in its prompt? Respond with a JSON object like {\"coder\": [\"f?\"], \"reviewer\": [\"f?\"]}. No other output.\n"
    }
  ],
  "target": "{\"coder\": [\"f5\", \"f6\", \"f7\", \"f1\"], \"reviewer\": [\"f5\", \"f3\", \"f4\"]}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task": "role_assignment",
    "evaluation_id": "pg_000__seed_1",
    "scenario_id": "pg_000",
    "shuffle_seed": 1,
    "roles": [
      "coder",
      "reviewer"
    ],
    "fragments": [
      {
        "id": "f1",
        "text": "After finding a bug, submit it to a **fresh** reviewer for review. Do not review your own report.\n\n**After the review returns:** execute all action steps in the reviewer's output.",
        "is_distractor": false
      },
      {
        "id": "f2",
        "text": "Claude responds well to clear, explicit instructions. Being specific about your desired output can help enhance results. If you want \"above and beyond\" behavior, explicitly request it rather than relying on the model to infer this from vague  ... [TRUNCATED 276 chars] ... w it. If they'd be confused, Claude will be too.\n\n- Be specific about the desired output format and constraints.\n- Provide instructions as sequential steps using numbered lists or bullet points when the order or completeness of steps matters.",
        "is_distractor": true
      },
      {
        "id": "f3",
        "text": "Audit the report you are given.",
        "is_distractor": false
      },
      {
        "id": "f4",
        "text": "Check the following dimensions one by one:\n\n| Dimension | Requirement |\n|-----------|-------------|\n| Scope | Within consensus-related code directories, or outside but directly affecting consensus logic |\n| Code references | Cited code snippe ... [TRUNCATED 1134 chars] ... ying victim code\n- Code references inconsistent with the latest testnet branch\n- PoC is incomplete and requires the reviewer to manually assemble it\n- Attack effect is equivalent to a Byzantine node simply going offline, with no amplification",
        "is_distractor": false
      },
      {
        "id": "f5",
        "text": "You are participating in a blockchain consensus-layer bug bounty contest.\n\nRead the contest rules first: ./contest-rules.md",
        "is_distractor": false
      },
      {
        "id": "f6",
        "text": "Find vulnerability in the code and write it up as a report.",
        "is_distractor": false
      },
      {
        "id": "f7",
        "text": "This project has been running stably for many years — any obvious fatal bug would have killed it long ago. So:\n\n- Don't report problems that are obvious at a glance (e.g. \"some map is never erased\") — if it were that obvious, the project team would have fixed it themselves\n- Truly valuable vulnerabilities hide in non-obvious interactions, races, and boundary conditions",
        "is_distractor": false
      }
    ],
    "distractor_id": "f2",
    "reference_need_sets": {
      "coder": [
        "f5",
        "f6",
        "f7",
        "f1"
      ],
      "reviewer": [
        "f5",
        "f3",
        "f4"
      ]
    }
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets perspective_gap_role_assignment \
    --limit 10  # 正式评测时删除这一行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['perspective_gap_role_assignment'],
    limit=10,  # 正式评测时删除这一行
)

run_task(task_cfg=task_cfg)
```
