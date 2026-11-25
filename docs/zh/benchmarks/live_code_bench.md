# Live-Code-Bench


## 概述

LiveCodeBench 是一个无数据污染的基准测试，用于在真实世界的竞赛编程问题上评估代码生成模型。它持续从编程平台收集新问题，以确保模型在训练期间未见过测试数据。

## 任务描述

- **任务类型**：竞赛编程 / 代码生成
- **输入**：包含输入/输出格式的编程问题描述
- **输出**：完整的解决方案代码
- **来源**：来自 LeetCode、Codeforces 和 AtCoder 的问题

## 主要特点

- 在模型训练截止日期后持续更新新问题
- 问题源自主要的竞赛编程平台
- 每个问题包含多个测试用例以进行全面评估
- 基于日期的过滤机制以控制数据污染
- 支持本地和沙箱代码执行

## 评估说明

- 通过 `TaskConfig.sandbox.engine='e2b'` 支持官方 E2B 和阿里云 Agent Sandbox，避免使用本地 Docker 和本地镜像下载。请参阅[沙箱指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html#e2b)。
- 默认配置使用 **0-shot** 评估
- 提示词遵循官方 LiveCodeBench 运行器：一个通用的“专家 Python 程序员”系统消息加上 `### Question` / `### Format` 用户消息
- **安全警告**：默认情况下，代码在本地环境中执行。我们建议使用沙箱执行。有关详细信息，请参阅[沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 使用 `start_date` 和 `end_date` 参数按日期过滤问题
- 默认超时时间为每个测试用例 6 秒
- 支持 `pass@k` 指标计算


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `live_code_bench` |
| **数据集ID** | [evalscope/livecodebench_code_generation_lite_parquet](https://modelscope.cn/datasets/evalscope/livecodebench_code_generation_lite_parquet/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `mean_and_pass_at_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,055 |
| 提示词长度（平均） | 1700.76 字符 |
| 提示词长度（最小/最大） | 717 / 4234 字符 |

## 样例示例

**子集**: `release_latest`

```json
{
  "input": [
    {
      "id": "6b6a6121",
      "content": "### Question:\nThere are three cards with letters $\\texttt{a}$, $\\texttt{b}$, $\\texttt{c}$ placed in a row in some order. You can do the following operation at most once: \n\n \n-  Pick two cards, and swap them.  Is it possible that the row becom ... [TRUNCATED] ... s from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n```python\n# YOUR CODE HERE\n```\n\n ### Answer: (use the provided format with backticks)\n\n"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "evaluation_sample": "{\"inputs\": [\"6\\nabc\\nacb\\nbac\\nbca\\ncab\\ncba\\n\", \"1\\nabc\\n\", \"3\\nabc\\nabc\\nabc\\n\", \"5\\ncab\\nacb\\ncba\\nbac\\nbca\\n\", \"6\\nabc\\nabc\\nabc\\nabc\\nabc\\nabc\\n\"], \"outputs\": [\"YES\\nYES\\nYES\\nNO\\nNO\\nYES\\n\", \"YES\\n\", \"YES\\nYES\\nYES\\n\", \"NO\\nYES\\nYES\\nYES\\nNO\\n\", \"YES\\nYES\\nYES\\nYES\\nYES\\nYES\\n\"], \"fn_name\": null}",
    "contest_date": "2023-08-21T00:00:00"
  }
}
```

*注意：为显示目的，部分内容已截断。*

## 提示模板

**系统提示：**
```text
You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.
```

**提示模板：**
```text
### Question:
{question_content}

{format_prompt} ### Answer: (use the provided format with backticks)


```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `start_date` | `str | null` | `None` | 从此日期开始过滤问题 (YYYY-MM-DD)。Null 保留所有。 |
| `end_date` | `str | null` | `None` | 过滤直到此日期的问题 (YYYY-MM-DD)。Null 保留所有。 |
| `debug` | `bool` | `False` | 启用详细调试日志记录并绕过某些安全检查。 |

## 沙箱配置

此基准测试需要沙箱环境进行代码执行。

```json
{
  "image": "python:3.11-slim",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  }
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets live_code_bench \
    --sandbox '{"enabled": true}' \
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
    datasets=['live_code_bench'],
    sandbox={'enabled': True},
    dataset_args={
        'live_code_bench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
