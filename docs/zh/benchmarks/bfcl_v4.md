# BFCL-v4


## 概述

BFCL-v4（Berkeley Function-Calling Leaderboard V4）是一个全面的基准测试，用于评估大语言模型（LLM）在智能体场景下的函数调用能力。它通过测试网络搜索、内存操作和格式敏感性等核心模块，为构建智能体应用提供基础能力评估。

## 任务描述

- **任务类型**：智能体函数调用评估
- **输入**：需要调用函数完成网络搜索、内存操作等任务的场景
- **输出**：带有正确参数的函数调用
- **能力范围**：网络搜索、内存读写、格式处理

## 核心特性

- 全面的智能体评估框架
- 测试 LLM 智能体的核心构建模块（搜索、内存、格式化）
- 多维度评分类别，支持细粒度分析
- 同时支持原生函数调用（FC）模型和非 FC 模型
- 可选集成 SerpAPI 实现网络搜索功能

## 评估说明

- **安装要求**：`pip install bfcl-eval==2025.10.27.1`
- 主要指标：各类别下的 **准确率（Accuracy）**
- 根据模型能力设置 `is_fc_model` 参数
- 可选：配置 `SERPAPI_API_KEY` 以启用网络搜索任务
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bfcl_v4` |
| **数据集ID** | [berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,106 |
| 提示词长度（平均） | 350.4 字符 |
| 提示词长度（最小/最大） | 36 / 10805 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `irrelevance` | 240 | 85.98 | 55 | 203 |
| `live_irrelevance` | 884 | 198.72 | 36 | 10805 |
| `live_multiple` | 1,053 | 149.16 | 42 | 4828 |
| `live_parallel` | 16 | 163.88 | 88 | 365 |
| `live_parallel_multiple` | 24 | 217.67 | 72 | 650 |
| `live_relevance` | 16 | 385.94 | 71 | 1830 |
| `live_simple` | 258 | 171.91 | 47 | 1247 |
| `memory_kv` | 155 | 646.46 | 585 | 826 |
| `memory_rec_sum` | 155 | 646.46 | 585 | 826 |
| `memory_vector` | 155 | 646.46 | 585 | 826 |
| `multi_turn_base` | 200 | 808.43 | 106 | 1756 |
| `multi_turn_long_context` | 200 | 808.41 | 106 | 1756 |
| `multi_turn_miss_func` | 200 | 811.88 | 110 | 1760 |
| `multi_turn_miss_param` | 200 | 863.88 | 167 | 1791 |
| `multiple` | 200 | 120.75 | 65 | 229 |
| `parallel` | 200 | 297.93 | 69 | 781 |
| `parallel_multiple` | 200 | 432.17 | 88 | 1249 |
| `simple_java` | 100 | 228.93 | 138 | 531 |
| `simple_javascript` | 50 | 222.34 | 118 | 637 |
| `simple_python` | 400 | 119.66 | 57 | 303 |
| `web_search_base` | 100 | 831.01 | 748 | 963 |
| `web_search_no_snippet` | 100 | 831.01 | 748 | 963 |

## 样例示例

**子集**：`irrelevance`

```json
{
  "input": [
    {
      "id": "4e05e910",
      "content": "[[{\"role\": \"user\", \"content\": \"Calculate the area of a triangle given the base is 10 meters and height is 5 meters.\"}]]"
    }
  ],
  "target": "{}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "irrelevance_0",
    "question": [
      [
        {
          "role": "user",
          "content": "Calculate the area of a triangle given the base is 10 meters and height is 5 meters."
        }
      ]
    ],
    "function": [
      {
        "name": "determine_body_mass_index",
        "description": "Calculate body mass index given weight and height.",
        "parameters": {
          "type": "dict",
          "properties": {
            "weight": {
              "type": "float",
              "description": "Weight of the individual in kilograms."
            },
            "height": {
              "type": "float",
              "description": "Height of the individual in meters."
            }
          },
          "required": [
            "weight",
            "height"
          ]
        }
      }
    ],
    "category": "irrelevance",
    "ground_truth": {}
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `underscore_to_dot` | `bool` | `True` | 在评估时将函数名中的下划线转换为点号。 |
| `is_fc_model` | `bool` | `True` | 指示被评估模型是否原生支持函数调用。 |
| `SERPAPI_API_KEY` | `str | null` | `None` | 启用 BFCL V4 中网络搜索功能的 SerpAPI 密钥。设为 null 则禁用网络搜索。 |

## 使用方法

### 通过 CLI 使用

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bfcl_v4 \
    --limit 10  # 正式评估时请移除此行
```

### 通过 Python 使用

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['bfcl_v4'],
    dataset_args={
        'bfcl_v4': {
            # subset_list: ['irrelevance', 'live_irrelevance', 'live_multiple']  # 可选，仅评估指定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```