# HumanEval


## 概述

HumanEval 是一个用于评估语言模型代码生成能力的基准测试。它包含 164 个手工编写的 Python 编程问题，每个问题都提供了函数签名、文档字符串（docstring）和完整的测试用例。

## 任务描述

- **任务类型**：代码生成（Python）
- **输入**：带有文档字符串的函数签名，描述期望的行为
- **输出**：完整的 Python 函数实现
- **语言**：仅限 Python

## 主要特性

- 包含 164 个精心设计的编程问题
- 每个问题均包含函数签名、文档字符串和测试用例
- 问题难度从简单的字符串操作到复杂的算法不等
- 提供标准参考解答
- 通过执行测试用例自动验证正确性

## 评估说明

- **安全警告**：默认情况下，代码将在本地环境中执行。我们强烈建议使用沙箱（sandbox）环境以确保安全。详情请参阅 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 支持 `pass@k` 指标计算，用于衡量生成质量
- 每个问题默认超时时间为 4 秒
- 若响应中包含 Markdown 代码块，则从中提取代码

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `humaneval` |
| **数据集 ID** | [opencompass/humaneval](https://modelscope.cn/datasets/opencompass/humaneval/summary) |
| **论文** | 无 |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `mean_and_pass_at_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 164 |
| 提示词长度（平均） | 609.6 字符 |
| 提示词长度（最小/最大） | 274 / 1519 字符 |

## 样例示例

**子集**: `openai_humaneval`

```json
{
  "input": [
    {
      "id": "5f652252",
      "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: f ... [TRUNCATED] ... Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
    }
  ],
  "target": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "HumanEval/0",
    "entry_point": "has_close_elements",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0 ... [TRUNCATED] ...  candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{question}
```

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

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets humaneval \
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
    datasets=['humaneval'],
    use_sandbox=True,
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```