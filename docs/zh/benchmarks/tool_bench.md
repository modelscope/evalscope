# ToolBench-Static


## 概述

ToolBench-Static 是一个用于评估 AI 模型在静态评测环境下使用工具和执行函数调用能力的基准测试。它通过与真实工具使用场景相符的任务，并结合标准答案进行对比，来衡量模型表现。

## 任务描述

- **任务类型**：工具使用 / 函数调用评估
- **输入**：包含可用工具和上下文的对话
- **输出**：正确的工具/函数调用序列
- **子集**：域内（已见工具）和域外（未见工具）

## 核心特性

- 静态评估（不实际执行工具）
- 同时测试工具选择和参数生成能力
- 包含域内和域外的评估划分
- 提供多种评估指标以实现全面评估
- 测试模型的规划与动作执行能力

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 使用五种指标：Act.EM、Plan.EM、F1、HalluRate、Rouge-L
- F1 分数为主要评估指标
- HalluRate 衡量工具幻觉（hallucination）频率
- 详细设置请参阅 [使用文档](https://evalscope.readthedocs.io/zh-cn/latest/third_party/toolbench.html)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tool_bench` |
| **数据集 ID** | [AI-ModelScope/ToolBench-Static](https://modelscope.cn/datasets/AI-ModelScope/ToolBench-Static/summary) |
| **论文** | N/A |
| **标签** | `FunctionCalling`, `Reasoning` |
| **指标** | `Act.EM`, `Plan.EM`, `F1`, `HalluRate`, `Rouge-L` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,369 |
| 提示词长度（平均） | 7628.39 字符 |
| 提示词长度（最小/最大） | 2720 / 23468 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `in_domain` | 1,588 | 7943.34 | 2720 | 23468 |
| `out_of_domain` | 781 | 6988.01 | 2788 | 20967 |

## 样例示例

**子集**: `in_domain`

```json
{
  "input": [
    {
      "id": "8b5b61b4",
      "content": "You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a fu ... [TRUNCATED] ...  \"enum\": [\"give_answer\", \"give_up_and_restart\"]}, \"final_answer\": {\"type\": \"string\", \"description\": \"The final answer you want to give the user. You should have this field if \\\"return_type\\\"==\\\"give_answer\\\"\"}}, \"required\": [\"return_type\"]}}]",
      "role": "system"
    },
    {
      "id": "97df9194",
      "content": "\nI'm planning a surprise birthday party for my sister and I need to find a suitable venue. Can you suggest event spaces in the city center with a capacity for 50 people? Also, recommend nearby catering services and provide me with a list of available dates for booking.\nBegin!\n",
      "role": "user"
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "target": "Thought: \nAction: spott\nAction Input: {\n  \"is_id\": \"city center\"\n}",
    "tools": [
      {
        "name": "get_place_by_my_ip_for_spott",
        "description": "This is the subfunction for tool \"spott\", you can use this tool.The description of this function is: \"Returns the place related to the IP where the request was performed. Returns \"Not Found\" error when no place is related to the IP.\"",
        "parameters": "{'type': 'object', 'properties': {}, 'required': [], 'optional': []}"
      },
      {
        "name": "autocomplete_places_for_spott",
        "description": "This is the subfunction for tool \"spott\", you can use this tool.The description of this function is: \"Returns a list of places matching a prefix and specified filter properties. Useful to create \"search as you type\" inputs.\"",
        "parameters": "{'type': 'object', 'properties': {}, 'required': [], 'optional': []}"
      },
      {
        "name": "get_place_by_ip_for_spott",
        "description": "This is the subfunction for tool \"spott\", you can use this tool.The description of this function is: \"Returns the Place where a given IP Address is located. Returns \"Not Found\" error when no place is related to the IP. When sending '127.0.0.1' or '0.0.0.0' IP Addresses it will return the Place from the request was performed.\"",
        "parameters": "{'type': 'object', 'properties': {'is_id': {'type': 'string', 'description': 'IP Address (v4 and v6 are supported).', 'example_value': '200.194.51.97'}, 'language': {'type': 'string', 'description': 'Specifies a language (ISO 639-1) to get the localized name of the place. If translation is not available, \"localizedName\" property will be null.'}}, 'required': ['is_id'], 'optional': ['language']}"
      },
      {
        "name": "webcams_list_orderby_order_sort_for_webcams_travel",
        "description": "This is the subfunction for tool \"webcams_travel\", you can use this tool.The description of this function is: \"This is a modifier. Returns the list of webcams ordered by {order}. The optional sorting direction is given by {sort}. Required {order}.\"",
        "parameters": "{'type': 'object', 'properties': {'sort': {'type': 'string', 'description': 'Possible values are: \"asc\" (ascending), or \"desc\" (descending).'}, 'order': {'type': 'string', 'description': 'Possible values are: \"popularity\" (default order: \"des ... [TRUNCATED] ...  the response. Possible values are: \"webcams\", \"categories\", \"continents\", \"countries\", \"regions\", \"properties\". Default is \"webcams\".', 'example_value': 'webcams:image,location'}}, 'required': ['sort', 'order'], 'optional': ['lang', 'show']}"
      },
      {
        "name": "webcams_list_region_region_region_for_webcams_travel",
        "description": "This is the subfunction for tool \"webcams_travel\", you can use this tool.The description of this function is: \"This is a modifier. Returns a list of webcams according to the listed region. Multiple regions must be separated by comma. Required: at least one {region}. Possible values are ISO 3166-1-alpha-2 codes for the country and a region code separated by a dot.\"",
        "parameters": "{'type': 'object', 'properties': {'region': {'type': 'string', 'description': 'Comma separated list of ISO 3166-1-alpha-2 codes for the country and a region code separated by a dot.'}, 'lang': {'type': 'string', 'description': 'Localize the r ... [TRUNCATED] ... sted in the response. Possible values are: \"webcams\", \"categories\", \"continents\", \"countries\", \"regions\", \"properties\". Default is \"webcams\".', 'example_value': 'webcams:image,location'}}, 'required': ['region'], 'optional': ['lang', 'show']}"
      },
      {
        "name": "webcams_list_bbox_ne_lat_ne_lng_sw_lat_sw_lng_for_webcams_travel",
        "description": "This is the subfunction for tool \"webcams_travel\", you can use this tool.The description of this function is: \"This is a modifier. Returns a list of the webcams in the bounding box given by north-east ({ne_lat},{ne_lng}) and south-west ({sw_lat},{sw_lng}) coordinates. Required: {ne_lat},{ne_lng},{sw_lat},{sw_lng}.\"",
        "parameters": "{'type': 'object', 'properties': {'ne_lat': {'type': 'integer', 'description': 'North-east WGS84 latitude of the bounding box.'}, 'sw_lng': {'type': 'integer', 'description': 'South-west WGS84 longitude of the bounding box.'}, 'sw_lat': {'typ ... [TRUNCATED] ...  values are: \"webcams\", \"categories\", \"continents\", \"countries\", \"regions\", \"properties\". Default is \"webcams\".', 'example_value': 'webcams:image,location'}}, 'required': ['ne_lat', 'sw_lng', 'sw_lat', 'ne_lng'], 'optional': ['lang', 'show']}"
      },
      {
        "name": "Finish",
        "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
        "parameters": "{'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}"
      }
    ],
    "messages": [
      {
        "role": "system",
        "content": "You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a fu ... [TRUNCATED] ...  \"enum\": [\"give_answer\", \"give_up_and_restart\"]}, \"final_answer\": {\"type\": \"string\", \"description\": \"The final answer you want to give the user. You should have this field if \\\"return_type\\\"==\\\"give_answer\\\"\"}}, \"required\": [\"return_type\"]}}]",
        "name": null
      },
      {
        "role": "user",
        "content": "\nI'm planning a surprise birthday party for my sister and I need to find a suitable venue. Can you suggest event spaces in the city center with a capacity for 50 people? Also, recommend nearby catering services and provide me with a list of available dates for booking.\nBegin!\n",
        "name": null
      }
    ]
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tool_bench \
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
    datasets=['tool_bench'],
    dataset_args={
        'tool_bench': {
            # subset_list: ['in_domain', 'out_of_domain']  # 可选，用于指定评估特定子集
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```