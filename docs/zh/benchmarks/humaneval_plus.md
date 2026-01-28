# HumanEvalPlus


## 概述

HumanEval Plus 是 OpenAI 的 HumanEval 基准测试的一个严格扩展版本，旨在解决代码生成评估中较高的假阳性率问题。它在原始测试用例的基础上，增加了数万个自动生成的输入，以暴露边界情况下的 bug 和功能错误。

## 任务描述

- **任务类型**：代码生成（Python）
- **输入**：包含文档字符串（docstring）的函数签名，描述预期行为
- **输出**：完整的 Python 函数实现
- **测试覆盖**：相比原始 HumanEval 大幅增强

## 主要特性

- 包含原始 HumanEval 的 164 个问题，并配备增强版测试套件
- 每个问题额外增加数万个测试用例
- 基于大语言模型（LLM）和变异（mutation）技术生成测试输入
- 能够发现原始 HumanEval 遗漏的边界情况和 bug
- 正确性评估更加严格且准确

## 评估说明

- 默认配置使用 **0-shot** 评估
- **安全警告**：默认情况下，代码将在本地环境中执行。我们强烈建议使用沙箱（sandbox）执行。详情请参阅 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 支持 `pass@k` 指标计算
- 默认超时时间为 300 秒，以适应庞大的测试套件
- 使用预装 numpy 的自定义 Docker 镜像

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `humaneval_plus` |
| **数据集ID** | [evalscope/humanevalplus](https://modelscope.cn/datasets/evalscope/humanevalplus/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `mean_and_pass_at_k` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 164 |
| 提示词长度（平均） | 609.57 字符 |
| 提示词长度（最小/最大） | 274 / 1519 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "dcff8346",
      "content": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: f ... [TRUNCATED] ... Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
    }
  ],
  "target": "\n\n    sorted_numbers = sorted(numbers)\n    for i in range(len(sorted_numbers) - 1):\n        if sorted_numbers[i + 1] - sorted_numbers[i] < threshold:\n            return True\n    return False\n\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "HumanEval/0",
    "entry_point": "has_close_elements",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "test": "\n\nimport numpy as np\n\ndef is_floats(x) -> bool:\n    # check if it is float; List[float]; Tuple[float]\n    if isinstance(x, float):\n        return True\n    if isinstance(x, (list, tuple)):\n        return all(isinstance(i, float) for i in x)\n   ... [TRUNCATED] ...  True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, False, True, True]\n    for i, (inp, exp) in enumerate(zip(inputs, results)):\n        assertion(candidate(*inp), exp, 0)\n"
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
  "image": "python3.11-numpy",
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
    --datasets humaneval_plus \
    --use-sandbox \
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
    datasets=['humaneval_plus'],
    use_sandbox=True,
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```