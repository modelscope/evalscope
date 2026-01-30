# MultiPL-E MBPP


## 概述

MultiPL-E MBPP 是一个源自 MBPP（Mostly Basic Python Programming）的多语言代码生成基准测试。它将原始 MBPP 扩展至 18 种编程语言，支持对代码生成能力进行跨语言评估。

## 任务描述

- **任务类型**：多语言代码生成
- **输入**：包含文档字符串（docstring）的编程问题提示
- **输出**：能通过测试用例的完整代码解决方案
- **语言**：18 种语言（C++、TypeScript、Shell、C#、Go、Java、Lua、JavaScript、PHP、Perl、Racket、R、Rust、Scala、Swift、Ruby、D、Julia）

## 主要特性

- 覆盖 18 种编程语言的多语言评估
- 基于执行的测试用例评估
- 支持代码生成的 pass@k 指标
- 使用 Docker 沙箱环境实现安全代码执行
- 源自 MBPP，保持一致的问题难度

## 评估说明

- **需要沙箱**：需使用沙箱环境以确保代码安全执行
- 默认使用 **test** 数据划分进行评估
- 主要指标：**准确率（Accuracy）**，采用 **pass@k** 聚合方式
- 每个测试用例超时时间：30 秒
- 沙箱环境配置详见 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `multiple_mbpp` |
| **数据集 ID** | [evalscope/MultiPL-E](https://modelscope.cn/datasets/evalscope/MultiPL-E/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `mean_and_pass_at_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 6,987 |
| 提示词长度（平均） | 379.32 字符 |
| 提示词长度（最小/最大） | 266 / 1002 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `mbpp-cpp` | 397 | 407.55 | 317 | 1002 |
| `mbpp-ts` | 390 | 359.57 | 295 | 674 |
| `mbpp-sh` | 382 | 362.53 | 296 | 696 |
| `mbpp-cs` | 386 | 549.09 | 481 | 867 |
| `mbpp-go` | 374 | 402.21 | 332 | 723 |
| `mbpp-java` | 386 | 553 | 474 | 889 |
| `mbpp-lua` | 397 | 332.52 | 277 | 651 |
| `mbpp-js` | 397 | 325.48 | 270 | 645 |
| `mbpp-php` | 397 | 336.17 | 280 | 655 |
| `mbpp-pl` | 396 | 337.87 | 282 | 657 |
| `mbpp-rkt` | 397 | 342.25 | 287 | 660 |
| `mbpp-r` | 397 | 326.42 | 271 | 644 |
| `mbpp-rs` | 354 | 351.9 | 285 | 669 |
| `mbpp-scala` | 396 | 434.01 | 364 | 750 |
| `mbpp-swift` | 396 | 352.64 | 285 | 667 |
| `mbpp-rb` | 397 | 321.34 | 266 | 641 |
| `mbpp-d` | 358 | 376.49 | 308 | 693 |
| `mbpp-jl` | 390 | 363.01 | 292 | 686 |

## 样例示例

**子集**: `mbpp-cpp`

```json
{
  "input": [
    {
      "id": "209e8d38",
      "content": "```cpp\n#include<assert.h>\n#include<bits/stdc++.h>\n// Write a cppthon function to identify non-prime numbers.\nbool is_not_prime(long n) {\n\n```\n\nPlease complete the above code according to the requirements in the docstring. Write the complete code and wrap it in markdown fenced code. The code should not contain `Main` function."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tests": "}\nint main() {\n    auto candidate = is_not_prime;\n    assert(candidate((2)) == (false));\n    assert(candidate((10)) == (true));\n    assert(candidate((35)) == (true));\n    assert(candidate((37)) == (false));\n}\n",
    "stop_tokens": [
      "\n}"
    ],
    "task_id": "mbpp_3_is_not_prime",
    "language": "cpp",
    "doctests": "transform"
  }
}
```

## 提示模板

**提示模板：**
```text
{prompt}
```

## 沙箱配置

此基准测试需要沙箱环境来执行代码。

```json
{
  "image": "volcengine/sandbox-fusion:server-20250609",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {},
    "multi_code_executor": {}
  },
  "memory_limit": "2g",
  "cpu_limit": "2.0"
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets multiple_mbpp \
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
    datasets=['multiple_mbpp'],
    use_sandbox=True,
    dataset_args={
        'multiple_mbpp': {
            # subset_list: ['mbpp-cpp', 'mbpp-ts', 'mbpp-sh']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```