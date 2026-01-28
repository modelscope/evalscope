# MBPP


## 概述

MBPP（Mostly Basic Python Problems）是一个包含约 1,000 个众包 Python 编程问题的基准测试，专为入门级程序员设计。它用于评估模型理解问题描述并生成正确 Python 代码的能力。

## 任务描述

- **任务类型**：代码生成（Python）
- **输入**：自然语言任务描述及测试用例
- **输出**：Python 函数实现
- **难度**：入门级编程问题

## 主要特点

- 包含约 1,000 个众包编程问题
- 覆盖编程基础和标准库使用
- 每个问题包含任务描述、参考解答和 3 个测试用例
- 问题设计为入门级程序员可解决
- 通过执行测试用例进行自动评估

## 评估说明

- 默认配置使用 **3-shot** 示例
- **安全警告**：必须在沙箱环境中安全执行代码。详情请参阅 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)。
- 支持 `pass@k` 指标计算
- 每个问题默认超时时间为 20 秒
- 若存在 `[BEGIN]...[DONE]` 块，则从中提取代码

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mbpp` |
| **数据集 ID** | [google-research-datasets/mbpp](https://modelscope.cn/datasets/google-research-datasets/mbpp/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 3-shot |
| **评估划分** | `test` |
| **训练划分** | `prompt` |
| **聚合方式** | `mean_and_pass_at_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 500 |
| 提示词长度（平均） | 1872.34 字符 |
| 提示词长度（最小/最大） | 1727 / 5896 字符 |

## 样例示例

**子集**: `full`

```json
{
  "input": [
    {
      "id": "f9c5e33a",
      "content": "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert ... [TRUNCATED] ... unction to remove first and last occurrence of a given character from the string. Your code should pass these tests:\n\nassert remove_Occ(\"hello\",\"l\") == \"heo\"\nassert remove_Occ(\"abcda\",\"a\") == \"bcd\"\nassert remove_Occ(\"PHP\",\"P\") == \"H\"\n[BEGIN]\n"
    }
  ],
  "target": "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "test_list": [
      "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
      "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
      "assert remove_Occ(\"PHP\",\"P\") == \"H\""
    ],
    "task_id": 11,
    "test_setup_code": ""
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

<details>
<summary>少样本模板</summary>

```text
You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:

assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
[BEGIN]
def similar_elements(test_tup1, test_tup2):
  res = tuple(set(test_tup1) & set(test_tup2))
  return (res)
[DONE]
You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
[BEGIN]
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
[DONE]
You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:

assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
[BEGIN]
import heapq as hq
def heap_queue_largest(nums,n):
  largest_nums = hq.nlargest(n, nums)
  return largest_nums
[DONE]
You are an expert Python programmer, and here is your task: {question} Your code should pass these tests:

{tests}
[BEGIN]

```

</details>

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
    --datasets mbpp \
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
    datasets=['mbpp'],
    use_sandbox=True,
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```