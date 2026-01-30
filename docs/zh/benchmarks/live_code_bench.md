# Live-Code-Bench

## 概述

LiveCodeBench 是一个无数据污染的基准测试，用于评估代码生成模型在真实世界竞赛编程问题上的表现。它持续从编程平台收集新问题，以确保模型在训练期间未见过测试数据。

## 任务描述

- **任务类型**：竞赛编程 / 代码生成
- **输入**：包含输入/输出格式的编程问题描述
- **输出**：完整的解决方案代码
- **来源**：LeetCode、Codeforces 和 AtCoder 的题目

## 主要特性

- 持续更新模型训练截止日期之后的新问题
- 题目来自主流竞赛编程平台
- 每道题包含多个测试用例，以进行全面评估
- 支持基于日期的过滤，防止数据污染
- 支持本地和沙箱环境中的代码执行

## 评估说明

- 默认配置使用 **0-shot** 评估
- **安全警告**：默认情况下，代码在本地环境中执行。我们建议使用沙箱执行。详情请参阅 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 使用 `start_date` 和 `end_date` 参数按日期筛选题目
- 每个测试用例默认超时时间为 6 秒
- 支持 `pass@k` 指标计算

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `live_code_bench` |
| **数据集ID** | [AI-ModelScope/code_generation_lite](https://modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |
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

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
### Question:
{question_content}

{format_prompt} ### Answer: (use the provided format with backticks)


```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `start_date` | `str | null` | `None` | 从该日期（YYYY-MM-DD）开始筛选题目。设为 null 则保留全部。 |
| `end_date` | `str | null` | `None` | 筛选截至该日期（YYYY-MM-DD）的题目。设为 null 则保留全部。 |
| `debug` | `bool` | `False` | 启用详细调试日志并绕过某些安全检查。 |

## 沙箱配置

此基准测试需要沙箱环境来执行代码。

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
    --use-sandbox \
    --limit 10  # 正式评估时请移除此行
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
    use_sandbox=True,
    dataset_args={
        'live_code_bench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```