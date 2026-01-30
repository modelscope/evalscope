# MultiPL-E HumanEval


## 概述

MultiPL-E HumanEval 是一个源自 OpenAI HumanEval 的多语言代码生成基准测试。它将原始 HumanEval 扩展至 18 种编程语言，支持对代码生成能力进行跨语言评估。

## 任务描述

- **任务类型**：多语言代码生成
- **输入**：包含函数签名和文档字符串的编程问题提示
- **输出**：能通过测试用例的完整函数实现
- **语言**：18 种语言（C++、TypeScript、Shell、C#、Go、Java、Lua、JavaScript、PHP、Perl、Racket、R、Rust、Scala、Swift、Ruby、D、Julia）

## 主要特性

- 覆盖 18 种编程语言的多语言评估
- 基于执行的测试用例评估
- 支持代码生成的 pass@k 指标
- 使用 Docker 沙箱环境确保安全执行
- 源自 HumanEval，保持一致的问题难度

## 评估说明

- **需要沙箱**：需使用沙箱环境以确保代码安全执行
- 默认使用 **test** 切分进行评估
- 主要指标：**Accuracy**，采用 **pass@k** 聚合方式
- 每个测试用例超时时间：30 秒
- 沙箱环境设置请参阅 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `multiple_humaneval` |
| **数据集ID** | [evalscope/MultiPL-E](https://modelscope.cn/datasets/evalscope/MultiPL-E/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估切分** | `test` |
| **聚合方式** | `mean_and_pass_at_k` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,864 |
| 提示词长度（平均） | 696.59 字符 |
| 提示词长度（最小/最大） | 285 / 2463 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `humaneval-cpp` | 161 | 797.74 | 351 | 2057 |
| `humaneval-ts` | 159 | 639.27 | 320 | 1529 |
| `humaneval-sh` | 158 | 648.14 | 325 | 1534 |
| `humaneval-cs` | 158 | 983.67 | 538 | 2335 |
| `humaneval-go` | 154 | 701.14 | 351 | 1616 |
| `humaneval-java` | 158 | 1009.53 | 531 | 2463 |
| `humaneval-lua` | 161 | 620.31 | 294 | 1497 |
| `humaneval-js` | 161 | 611.34 | 287 | 1490 |
| `humaneval-php` | 161 | 632.01 | 298 | 1551 |
| `humaneval-pl` | 161 | 611.27 | 296 | 1481 |
| `humaneval-rkt` | 161 | 634.22 | 304 | 1537 |
| `humaneval-r` | 161 | 607.78 | 285 | 1484 |
| `humaneval-rs` | 156 | 686.67 | 313 | 1591 |
| `humaneval-scala` | 160 | 832.81 | 423 | 1998 |
| `humaneval-swift` | 158 | 660.88 | 323 | 1556 |
| `humaneval-rb` | 161 | 611.55 | 289 | 1474 |
| `humaneval-d` | 156 | 640.72 | 328 | 1501 |
| `humaneval-jl` | 159 | 616.4 | 303 | 1478 |

## 样例示例

**子集**: `humaneval-cpp`

```json
{
  "input": [
    {
      "id": "8f096947",
      "content": "```cpp\n#include<assert.h>\n#include<bits/stdc++.h>\n// Check if in given vector of numbers, are any two numbers closer to each other than\n// given threshold.\n// >>> has_close_elements((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.0f}) ... [TRUNCATED] ... ents(std::vector<float> numbers, float threshold) {\n\n```\n\nPlease complete the above code according to the requirements in the docstring. Write the complete code and wrap it in markdown fenced code. The code should not contain `Main` function."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tests": "}\nint main() {\n    auto candidate = has_close_elements;\n    assert(candidate((std::vector<float>({(float)1.0f, (float)2.0f, (float)3.9f, (float)4.0f, (float)5.0f, (float)2.2f})), (0.3f)) == (true));\n    assert(candidate((std::vector<float>({( ... [TRUNCATED] ... (std::vector<float>({(float)1.1f, (float)2.2f, (float)3.1f, (float)4.1f, (float)5.1f})), (1.0f)) == (true));\n    assert(candidate((std::vector<float>({(float)1.1f, (float)2.2f, (float)3.1f, (float)4.1f, (float)5.1f})), (0.5f)) == (false));\n}\n",
    "stop_tokens": [
      "\n}"
    ],
    "task_id": "HumanEval_0_has_close_elements",
    "language": "cpp",
    "doctests": "transform"
  }
}
```

*注：部分内容因展示需要已截断。*

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
    --datasets multiple_humaneval \
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
    datasets=['multiple_humaneval'],
    use_sandbox=True,
    dataset_args={
        'multiple_humaneval': {
            # subset_list: ['humaneval-cpp', 'humaneval-ts', 'humaneval-sh']  # 可选，用于指定评估特定子集
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```