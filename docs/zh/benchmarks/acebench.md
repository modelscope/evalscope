# ACEBench


## 概述

ACEBench 评估大语言模型在真实场景中使用工具的能力：选择正确的 API、填充参数、对无法满足的请求进行合理拒绝，以及在模拟环境中驱动多步骤智能体任务。数据分为三类——`normal`（常规工具使用）、`special`（不完整、错误或超出范围的请求）和 `agent`（多步骤、多轮交互）——并在 17 个细粒度类别上进行报告。

## 任务描述

- **任务类型**：函数调用与智能体工具使用
- **输入**：对话历史、API 规范，以及可选的时间或角色档案上下文
- **输出**：一个 `[ApiName(key='value')]` 调用列表、一句诊断说明，或完整的智能体轨迹
- **领域**：涵盖技术、金融、健康和社会等 8 个主领域及 68 个子领域

## 主要特性

- 包含 1023 个英文样本和 1017 个中文样本，可通过 `extra_params.language` 参数选择。
- 使用官方 ACEBench 提示词和官方 `[ApiName(...)]` 输出格式；若输出无法被正确解析，则得分为零，不会通过宽松解析进行挽救。
- `normal_multi_turn_*` 类别按对话评分：只有所有步骤都正确，该对话才算正确，符合官方的回合级聚合方式。
- `agent` 类别会在 ACEBench 模拟的手机、外卖和旅行 API 环境中实际运行，并根据最终环境状态进行评分。

## 评估说明

- `acc` 是主要指标。对于 `normal` 和 `special` 类别，它表示答案准确率；对于 `agent` 类别，它表示最终状态准确率。`process_acc` 额外报告 `agent` 样本的关键里程碑进展，以及 `normal_multi_turn_*` 样本的每步进展。
- 报告包含官方分组（ATOM、SINGLE_TURN、MULTI_TURN、NORMAL、SPECIAL、AGENT）和一个总体（OVERALL）得分，权重为 `normal` 0.578 / `special` 0.2676 / `agent` 0.1545。权重会根据实际评估的分组重新归一化，因此部分运行结果仍具可解释性。
- `agent_multi_turn` 类别还需要一个用户模拟器；请通过 `extra_params.user_model` 指定扮演用户的模型（官方运行器使用 `gpt-4o`）。若未配置，这些 rollout 将失败并得分为零，因此在查看 OVERALL 分数前请务必配置此项。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `acebench` |
| **数据集ID** | [evalscope/acebench](https://modelscope.cn/datasets/evalscope/acebench/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **指标** | `acc`, `process_acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `normal` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,023 |
| 提示词长度（平均） | 6032.98 字符 |
| 提示词长度（最小/最大） | 2295 / 11835 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `normal_single_turn_single_function` | 100 | 5165.79 | 2461 | 9553 |
| `normal_single_turn_parallel_function` | 100 | 5036.21 | 2295 | 9644 |
| `normal_multi_turn_user_adjust` | 123 | 4658.51 | 3172 | 6976 |
| `normal_multi_turn_user_switch` | 100 | 7546.46 | 3467 | 11835 |
| `normal_similar_api` | 50 | 3511.84 | 2484 | 6209 |
| `normal_preference` | 50 | 8637.66 | 7107 | 10381 |
| `normal_atom_bool` | 50 | 7377.62 | 4762 | 9727 |
| `normal_atom_enum` | 50 | 7676.94 | 4927 | 11337 |
| `normal_atom_number` | 50 | 7481.46 | 4851 | 10278 |
| `normal_atom_list` | 50 | 7524.06 | 4910 | 10514 |
| `normal_atom_object_deep` | 50 | 6102.02 | 2873 | 9755 |
| `normal_atom_object_short` | 50 | 5139.5 | 2343 | 8921 |
| `special_incomplete` | 50 | 6177.34 | 3473 | 10806 |
| `special_error_param` | 50 | 4499.78 | 3121 | 6090 |
| `special_irrelevant` | 50 | 6011.94 | 3778 | 8492 |
| `agent_multi_step` | 20 | 6407.9 | 6343 | 6472 |
| `agent_multi_turn` | 30 | 6290.97 | 5505 | 6630 |

## 样例示例

**子集**: `normal_single_turn_single_function`

```json
{
  "input": [
    {
      "id": "9198db95",
      "content": "You are an AI assistant with the role name \"assistant.\" Based on the provided API specifications and conversation history from steps 1 to t, generate the API requests that the assistant should call in step t+1. The API requests should be outp ... [TRUNCATED 3788 chars] ... '}, 'effects': {'description': 'List of audio effects to apply.', 'type': 'array', 'items': {'type': 'string', 'enum': ['reverb', 'echo', 'distortion']}}}, 'required': ['frequency', 'gain']}}}, 'required': ['microphone', 'performanceTime']}}]"
    },
    {
      "id": "61cfd720",
      "content": "Conversation history 1..t:\nuser: I have been fascinated recently with total solar eclipses. I am planning my next travel and would like to know when the next total solar eclipse will be visible in Greece, specifically in Athens, over the next five years.\n"
    }
  ],
  "target": "{\"ground_truth\": {\"NightSkyAnalysis_performEclipseAnalysis\": {\"dateRange\": {\"startDate\": \"2023-01-01\", \"endDate\": \"2028-01-01\"}, \"location\": {\"latitude\": 37.9838, \"longitude\": 23.7275}, \"eclipseType\": \"total\"}}, \"mile_stone\": []}",
  "id": 0,
  "group_id": 0,
  "subset_key": "normal_single_turn_single_function",
  "metadata": {
    "id": "normal_single_turn_single_function_0",
    "test_category": "normal_single_turn_single_function",
    "dialogue_id": "normal_single_turn_single_function_0",
    "language": "en",
    "functions": [
      {
        "name": "NightSkyAnalysis_performEclipseAnalysis",
        "description": "Analyzes the occurrence of solar eclipses, categorizes them into types, and predicts future occurrences based on historical data and celestial mechanics.",
        "parameters": {
          "type": "object",
          "properties": {
            "dateRange": {
              "description": "The range of dates for which to analyze solar eclipses.",
              "type": "object",
              "properties": {
                "startDate": {
                  "description": "The starting date for the analysis in YYYY-MM-DD format.",
                  "type": "string"
                },
                "endDate": {
                  "description": "The ending date for the analysis in YYYY-MM-DD format.",
                  "type": "string"
                }
              },
              "required": [
                "startDate",
                "endDate"
              ]
            },
            "location": {
              "description": "Geographical coordinates to focus the eclipse analysis.",
              "type": "object",
              "properties": {
                "latitude": {
                  "description": "Latitude of the location.",
                  "type": "number",
                  "minimum": -90,
                  "maximum": 90
                },
                "longitude": {
                  "description": "Longitude of the location.",
                  "type": "number",
                  "minimum": -180,
                  "maximum": 180
                }
              },
              "required": [
                "latitude",
                "longitude"
              ]
            },
            "eclipseType": {
              "description": "The type of solar eclipse to specifically analyze.",
              "type": "string",
              "enum": [
                "total",
                "annular",
                "partial"
              ]
            }
          },
          "required": [
            "dateRange",
            "location"
          ]
        }
      },
      {
        "name": "AudioPerformanceOptimizer_optimizeMicrophoneSettings",
        "description": "Optimizes microphone settings for live performances, focusing on dynamic microphones to enhance sound quality and reduce feedback.",
        "parameters": {
          "type": "object",
          "properties": {
            "microphone": {
              "description": "Details of the microphone used.",
              "type": "object",
              "properties": {
                "type": {
                  "description": "Type of the microphone.",
                  "type": "string",
                  "enum": [
                    "dynamic",
                    "condenser",
                    "ribbon"
                  ]
                },
                "model": {
                  "description": "Model of the microphone.",
                  "type": "string"
                }
              },
              "required": [
                "type",
                "model"
              ]
            },
            "performanceTime": {
              "description": "Scheduled time for the performance.",
              "type": "string",
              "enum": [
                "morning",
                "afternoon",
                "evening",
                "night"
              ]
            },
            "environment": {
              "description": "Environmental conditions of the performance area.",
              "type": "object",
              "properties": {
                "humidity": {
                  "description": "Humidity level as a percentage.",
                  "type": "integer",
                  "minimum": 0,
                  "maximum": 100
                },
                "temperature": {
                  "description": "Temperature in Celsius.",
                  "type": "integer"
                }
              }
            },
            "soundSettings": {
              "description": "Specific sound settings to apply.",
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "frequency": {
                    "description": "Frequency adjustments in Hz.",
                    "type": "integer"
                  },
                  "gain": {
                    "description": "Gain adjustments in dB.",
                    "type": "integer"
                  },
                  "effects": {
                    "description": "List of audio effects to apply.",
                    "type": "array",
                    "items": {
                      "type": "string",
                      "enum": [
                        "reverb",
                        "echo",
                        "distortion"
                      ]
                    }
                  }
                },
                "required": [
                  "frequency",
                  "gain"
                ]
              }
            }
          },
          "required": [
            "microphone",
            "performanceTime"
          ]
        }
      }
    ],
    "ground_truth": {
      "NightSkyAnalysis_performEclipseAnalysis": {
        "dateRange": {
          "startDate": "2023-01-01",
          "endDate": "2028-01-01"
        },
        "location": {
          "latitude": 37.9838,
          "longitude": 23.7275
        },
        "eclipseType": "total"
      }
    },
    "mile_stone": [],
    "initial_config": {},
    "involved_classes": [],
    "question": "user: I have been fascinated recently with total solar eclipses. I am planning my next travel and would like to know when the next total solar eclipse will be visible in Greece, specifically in Athens, over the next five years.\n",
    "time": "The current time is January 01, 2023, Sunday",
    "profile": ""
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `language` | `str` | `en` | 要评估的数据集语言，可选 `en` 或 `zh`。 |
| `user_model` | `str` | `` | 在 `agent_multi_turn` rollout 中扮演用户的模型，例如 `gpt-4o`。若未设置，这些 rollout 将失败并得分为零。 |
| `user_model_api_url` | `str` | `` | `user_model` 的基础 URL，默认为 `MODELSCOPE_API_BASE`。 |
| `user_model_api_key` | `str` | `` | `user_model` 的 API 密钥，默认为 `MODELSCOPE_SDK_TOKEN`。 |
| `max_dialog_turns` | `int` | `40` | 智能体 rollout 的最大步数。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets acebench \
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
    datasets=['acebench'],
    dataset_args={
        'acebench': {
            # subset_list: ['normal_single_turn_single_function', 'normal_single_turn_parallel_function', 'normal_multi_turn_user_adjust']  # 可选，评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
