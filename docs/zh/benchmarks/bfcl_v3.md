# BFCL-v3


## 概述

BFCL（Berkeley Function Calling Leaderboard）v3 是首个全面且可执行的函数调用评估基准，用于评估大语言模型（LLM）调用函数的能力。它评估了多种形式的函数调用、多样化的场景以及可执行性。

## 任务描述

- **任务类型**：函数调用 / 工具使用评估
- **输入**：包含可用函数定义的用户查询
- **输出**：带有正确参数的函数调用
- **类别**：AST_NON_LIVE、AST_LIVE、RELEVANCE、MULTI_TURN

## 主要特性

- 全面的函数调用评估
- 测试简单、多重和并行函数调用
- 多语言支持（Python、Java、JavaScript）
- 相关性检测（处理无关查询）
- 多轮对话中的函数调用
- 提供实时 API 端点用于执行测试

## 评估说明

- 评估前需安装 `pip install bfcl-eval==2025.10.27.1`
- 对于原生支持函数调用的模型，请设置 `is_fc_model=True`
- `underscore_to_dot` 参数控制函数名称的格式化方式
- 详细设置请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v3.html)
- 结果按类别（AST、相关性、多轮）细分

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bfcl_v3` |
| **数据集ID** | [AI-ModelScope/bfcl_v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,441 |
| 提示词长度（平均） | 291.25 字符 |
| 提示词长度（最小/最大） | 36 / 10805 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `simple` | 400 | 119.66 | 57 | 303 |
| `multiple` | 200 | 120.75 | 65 | 229 |
| `parallel` | 200 | 298.92 | 69 | 781 |
| `parallel_multiple` | 200 | 432.17 | 88 | 1249 |
| `java` | 100 | 228.93 | 138 | 531 |
| `javascript` | 50 | 222.34 | 118 | 637 |
| `live_simple` | 258 | 161.74 | 47 | 1247 |
| `live_multiple` | 1,053 | 138.76 | 42 | 4828 |
| `live_parallel` | 16 | 150.44 | 88 | 365 |
| `live_parallel_multiple` | 24 | 217.67 | 72 | 650 |
| `irrelevance` | 240 | 85.96 | 55 | 203 |
| `live_relevance` | 18 | 263.83 | 71 | 1214 |
| `live_irrelevance` | 882 | 188.26 | 36 | 10805 |
| `multi_turn_base` | 200 | 803.33 | 106 | 1756 |
| `multi_turn_miss_func` | 200 | 806.75 | 110 | 1760 |
| `multi_turn_miss_param` | 200 | 858.2 | 167 | 1791 |
| `multi_turn_long_context` | 200 | 803.29 | 106 | 1756 |

## 样例示例

**子集**：`simple`

```json
{
  "input": [
    {
      "id": "15e84989",
      "content": "[[{\"role\": \"user\", \"content\": \"Find the area of a triangle with a base of 10 units and height of 5 units.\"}]]"
    }
  ],
  "target": "[{\"calculate_triangle_area\": {\"base\": [10], \"height\": [5], \"unit\": [\"units\", \"\"]}}]",
  "id": 0,
  "group_id": 0,
  "subset_key": "simple",
  "metadata": {
    "id": "simple_0",
    "multi_turn": false,
    "functions": [
      {
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle given its base and height.",
        "parameters": {
          "type": "dict",
          "properties": {
            "base": {
              "type": "integer",
              "description": "The base of the triangle."
            },
            "height": {
              "type": "integer",
              "description": "The height of the triangle."
            },
            "unit": {
              "type": "string",
              "description": "The unit of measure (defaults to 'units' if not specified)"
            }
          },
          "required": [
            "base",
            "height"
          ]
        }
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculate_triangle_area",
          "description": "Calculate the area of a triangle given its base and height. Note that the provided function is in Python 3 syntax.",
          "parameters": {
            "type": "object",
            "properties": {
              "base": {
                "type": "integer",
                "description": "The base of the triangle."
              },
              "height": {
                "type": "integer",
                "description": "The height of the triangle."
              },
              "unit": {
                "type": "string",
                "description": "The unit of measure (defaults to 'units' if not specified)"
              }
            },
            "required": [
              "base",
              "height"
            ]
          }
        }
      }
    ],
    "missed_functions": "{}",
    "initial_config": {},
    "involved_classes": [],
    "turns": [
      [
        {
          "role": "user",
          "content": "Find the area of a triangle with a base of 10 units and height of 5 units."
        }
      ]
    ],
    "language": "Python",
    "test_category": "simple",
    "subset": "simple",
    "ground_truth": [
      {
        "calculate_triangle_area": {
          "base": [
            10
          ],
          "height": [
            5
          ],
          "unit": [
            "units",
            ""
          ]
        }
      }
    ],
    "should_execute_tool_calls": false,
    "missing_functions": {},
    "is_fc_model": true
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `underscore_to_dot` | `bool` | `True` | 在评估时将函数名中的下划线转换为点号。 |
| `is_fc_model` | `bool` | `True` | 表示被评估的模型原生支持函数调用。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bfcl_v3 \
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
    datasets=['bfcl_v3'],
    dataset_args={
        'bfcl_v3': {
            # subset_list: ['simple', 'multiple', 'parallel']  # 可选，评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```