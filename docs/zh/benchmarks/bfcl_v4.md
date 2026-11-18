# BFCL-v4


## 概述

BFCL-v4（伯克利函数调用排行榜 V4）是一个综合基准测试，用于评估大语言模型（LLM）的智能体函数调用能力。它测试了作为智能体应用构建模块的网络搜索、内存操作和格式敏感性。

## 任务描述

- **任务类型**：智能体函数调用评估
- **输入**：需要函数调用的场景，如网络搜索、内存操作等
- **输出**：带有参数的函数调用以及智能体任务的最终答案
- **能力**：网络搜索、内存读写、格式处理

## 主要特点

- 全面的智能体评估框架
- 测试 LLM 智能体的构建模块（搜索、内存、格式化）
- 提供多个评分类别以进行详细分析
- 支持 FC 模型和非 FC 模型
- 可选的网络搜索功能，集成 SerpAPI

## 评估说明

- **安装要求**：`pip install 'evalscope[bfcl]'`，需配合 `bfcl-eval==2026.3.23`
- 主要指标：**Accuracy**（按类别统计）
- 根据模型能力配置 `is_fc_model`
- 内存任务会准备所选样本所需的场景。如果内存快照为空，将记录警告
- 网络搜索需要有效的 `SERPAPI_API_KEY`。若要在没有密钥的情况下进行评估，请从 `subset_list` 中省略 `web_search_base` 和 `web_search_no_snippet`
- [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/third_party/bfcl_v4.html)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bfcl_v4` |
| **数据集ID** | [berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling` |
| **指标** | `accuracy` |
| **默认 Shots** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,106 |
| 提示词长度（平均值） | 348.78 字符 |
| 提示词长度（最小/最大值） | 36 / 10805 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
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
| `multi_turn_base` | 200 | 798.28 | 106 | 1756 |
| `multi_turn_long_context` | 200 | 798.24 | 106 | 1756 |
| `multi_turn_miss_func` | 200 | 801.57 | 110 | 1760 |
| `multi_turn_miss_param` | 200 | 853.16 | 167 | 1791 |
| `multiple` | 200 | 120.75 | 65 | 229 |
| `parallel` | 200 | 297.93 | 69 | 781 |
| `parallel_multiple` | 200 | 432.17 | 88 | 1249 |
| `simple_java` | 100 | 228.93 | 138 | 531 |
| `simple_javascript` | 50 | 222.34 | 118 | 637 |
| `simple_python` | 400 | 119.66 | 57 | 303 |
| `web_search_base` | 100 | 831.01 | 748 | 963 |
| `web_search_no_snippet` | 100 | 831.01 | 748 | 963 |

## 样例示例

**子集**: `irrelevance`

```json
{
  "input": [
    {
      "id": "9f29fc22",
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
| `underscore_to_dot` | `bool` | `True` | 在评估过程中将函数名中的下划线转换为点号。 |
| `is_fc_model` | `bool` | `True` | 指示被评估的模型原生支持函数调用。 |
| `SERPAPI_API_KEY` | `str \| null` | `None` | 用于网络搜索的 SerpAPI 密钥。如果未设置，BFCL 将读取环境变量 SERPAPI_API_KEY。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bfcl_v4 \
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
    datasets=['bfcl_v4'],
    dataset_args={
        'bfcl_v4': {
            # subset_list: ['irrelevance', 'live_irrelevance', 'live_multiple']  # 可选，用于评估特定子集
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
