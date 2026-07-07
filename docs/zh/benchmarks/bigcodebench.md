# BigCodeBench


## 概述

BigCodeBench 是一个易于使用的基准测试，用于通过代码解决实际且具有挑战性的任务。它在更贴近现实的场景中评估大语言模型（LLMs）的真实编程能力，涵盖来自 139 个流行库的 723 个 API 调用。

## 任务描述

- **任务类型**：代码生成（Python）
- **输入**：编程任务描述（文档字符串或自然语言指令）
- **输出**：完整的 Python 函数实现
- **库**：139 个流行的 Python 库（如 numpy、pandas、sklearn 等）

## 主要特性

- 包含 1,140 个上下文丰富的 Python 编程任务
- 提供两种评估模式：Complete（基于文档字符串）和 Instruct（基于自然语言指令）
- 覆盖来自 139 个流行库的多样化函数调用
- 使用 `unittest.TestCase` 进行全面的正确性验证
- 支持 pass@k 指标计算

## 评估说明

- **需要沙箱环境**：要求预装 70 多个 Python 库的沙箱环境
- 通过 `split` 参数选择两种模式：`complete`（文档字符串补全）或 `instruct`（自然语言指令）
- 每个问题默认超时时间为 240 秒
- 启用 `calibrate` 选项会在生成代码前添加 `code_prompt`，以对齐函数签名
- 沙箱环境设置详见 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bigcodebench` |
| **数据集ID** | [evalscope/bigcodebench](https://modelscope.cn/datasets/evalscope/bigcodebench/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分版本** | `v0.1.4` |
| **聚合方式** | `mean_and_pass_at_k` |


## 数据统计

*统计数据暂不可用。*

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "657d2473",
      "content": "Calculates the average of the sums of absolute differences between each pair of consecutive numbers for all permutations of a given list. Each permutation is shuffled before calculating the differences. Args: - numbers (list): A list of numbe ... [TRUNCATED 75 chars] ... loat: The average of the sums of absolute differences for each shuffled permutation of the list.\nYou should write self-contained code starting with:\n```\nimport itertools\nfrom random import shuffle\ndef task_func(numbers=list(range(1, 3))):\n```"
    }
  ],
  "target": "    permutations = list(itertools.permutations(numbers))\n    sum_diffs = 0\n\n    for perm in permutations:\n        perm = list(perm)\n        shuffle(perm)\n        diffs = [abs(perm[i] - perm[i+1]) for i in range(len(perm)-1)]\n        sum_diffs += sum(diffs)\n\n    avg_sum_diffs = sum_diffs / len(permutations)\n    \n    return avg_sum_diffs",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "BigCodeBench/0",
    "entry_point": "task_func",
    "complete_prompt": "import itertools\nfrom random import shuffle\n\ndef task_func(numbers=list(range(1, 3))):\n    \"\"\"\n    Calculates the average of the sums of absolute differences between each pair of consecutive numbers \n    for all permutations of a given list.  ... [TRUNCATED 187 chars] ... age of the sums of absolute differences for each shuffled permutation of the list.\n\n    Requirements:\n    - itertools\n    - random.shuffle\n\n    Example:\n    >>> result = task_func([1, 2, 3])\n    >>> isinstance(result, float)\n    True\n    \"\"\"\n",
    "code_prompt": "import itertools\nfrom random import shuffle\ndef task_func(numbers=list(range(1, 3))):\n",
    "test": "import unittest\nfrom unittest.mock import patch\nfrom random import seed, shuffle\nimport itertools\nclass TestCases(unittest.TestCase):\n    def test_default_numbers(self):\n        # Test with default number range (1 to 10) to check that the res ... [TRUNCATED 2578 chars] ...  x: seed(1) or shuffle(x)):\n            result1 = task_func([1, 2, 3])\n        with patch('random.shuffle', side_effect=lambda x: seed(1) or shuffle(x)):\n            result2 = task_func([1, 2, 4])\n        self.assertNotEqual(result1, result2)"
  }
}
```

## 提示模板

**提示模板：**
```text
{prompt}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `split` | `str` | `instruct` | 评估模式："complete"（文档字符串补全）或 "instruct"（自然语言指令）。可选值：['complete', 'instruct'] |
| `version` | `str` | `default` | 数据集版本。使用 "default" 表示最新可用版本。 |
| `calibrate` | `bool` | `True` | 是否在解决方案前添加 `code_prompt` 以对齐函数签名。 |
| `docker_build_context` | `str` | `` | 可选的本地 Docker 构建上下文。设置后将覆盖默认沙箱镜像。 |
| `dockerfile` | `str` | `Dockerfile` | `docker_build_context` 内的 Dockerfile 路径。 |
| `force_rebuild` | `bool` | `False` | 是否强制重建可选的本地 Docker 镜像。 |

## 沙箱配置

此基准测试需要沙箱环境来执行代码。

```json
{
  "image": "bigcodebench/bigcodebench-evaluate:latest",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  },
  "memory_limit": "4g"
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bigcodebench \
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
    datasets=['bigcodebench'],
    sandbox={'enabled': True},
    dataset_args={
        'bigcodebench': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```