# MBPP-Plus

## 概述

MBPP Plus 是 MBPP 基准测试的增强版本，旨在提升对基础 Python 编程合成任务的评估可靠性。它解决了原始数据集中的质量问题，并显著增加了每个问题的测试覆盖范围。

## 任务描述

- **任务类型**：代码生成（Python）
- **输入**：包含测试断言的编程任务描述
- **输出**：Python 函数实现
- **改进点**：经过清洗的数据与扩展的测试套件

## 主要特性

- 修正了参考代码并澄清了问题描述
- 每个问题大幅扩充了测试用例（通过自动化生成）
- 采用基于大语言模型（LLM）和基于变异（mutation-based）的测试生成策略
- 能够暴露原始 MBPP 中遗漏的边界情况错误
- 提供更严格、更准确的代码正确性评估

## 评估说明

- 默认配置使用 **0-shot** 评估
- **安全警告**：默认情况下，代码将在本地环境中执行。我们强烈建议使用沙箱环境执行。详情请参阅 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 支持 `pass@k` 指标计算
- 每个问题默认超时时间为 20 秒
- 评估标准比原始 MBPP 基准测试更为严格

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mbpp_plus` |
| **数据集ID** | [evalscope/mbppplus](https://modelscope.cn/datasets/evalscope/mbppplus/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |
| **聚合方式** | `mean_and_pass_at_k` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 378 |
| 提示词长度（平均） | 375.53 字符 |
| 提示词长度（最小/最大） | 222 / 2801 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "aa126494",
      "content": "You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists. Your code should pass these tests:\n\nassert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\nassert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))\nassert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))"
    }
  ],
  "target": "\ndef similar_elements(test_tup1, test_tup2):\n  return tuple(set(test_tup1) & set(test_tup2))\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": 2,
    "source_file": "Benchmark Questions Verification V2.ipynb",
    "test_imports": [],
    "test_list": [
      "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
      "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
      "assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))"
    ],
    "test": "import numpy as np\nfrom math import inf\n\ndef is_floats(x) -> bool:\n    # check if it is float; List[float]; Tuple[float]\n    if isinstance(x, float):\n        return True\n    if isinstance(x, (list, tuple)):\n        return all(isinstance(i, fl ... [TRUNCATED] ... wvS', 'tUqF', 'ITntCqEvPi', 'kx'), (1, 2, 3, 4, 5, 6, 7, 8, 9), (11, 12, 13, 14, 15, 16, 17, 19, 20), (19, 5), (1, 2, 3, 6, 7, 8, 9, 10, 12)]\nfor i, (inp, exp) in enumerate(zip(inputs, results)):\n    assertion(similar_elements(*inp), exp, 0)\n"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
```

## 沙箱配置

此基准测试需要在沙箱环境中执行代码。

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

### 通过命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mbpp_plus \
    --use-sandbox \
    --limit 10  # 正式评估时请移除此行
```

### 通过 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['mbpp_plus'],
    use_sandbox=True,
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```