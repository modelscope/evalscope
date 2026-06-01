# ACEBench


## 概述

ACEBench 是一个工具使用基准测试，用于评估大语言模型是否能够选择 API、填充参数、处理异常请求，并完成真实的智能体任务。

## 任务描述

- **任务类型**：函数调用与智能体工具使用
- **输入**：对话历史、API 规范、可选的时间/用户资料上下文，以及智能体任务上下文
- **输出**：函数调用或特殊情况下的诊断文本
- **子集**：normal（常规）、special（特殊）、agent（智能体）

## 评估说明

- 适配器将 ACEBench 的 API 规范作为 EvalScope 工具传入，同时也为纯文本模型提供简洁的文本指令。
- 常规样本通过匹配函数名称和参数进行评分。
- 特殊样本根据 ACEBench 的诊断文本规范进行评分。
- 智能体样本根据 ACEBench 的里程碑报告 `process_acc`。如果模型返回最终状态的 JSON 对象，则还会报告 `end_state_acc` 并将其作为 `acc`；否则 `acc` 与 `process_acc` 一致。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `acebench` |
| **数据集ID** | [evalscope/acebench](https://modelscope.cn/datasets/evalscope/acebench/summary) |
| **论文** | N/A |
| **标签** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **指标** | `acc`, `process_acc`, `end_state_acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `normal` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,023 |
| 提示词长度（平均） | 4676.74 字符 |
| 提示词长度（最小/最大） | 1007 / 10555 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `normal` | 823 | 4859.68 | 1070 | 10555 |
| `special` | 150 | 3449.69 | 1007 | 8692 |
| `agent` | 50 | 5346.72 | 4454 | 5656 |

## 样例示例

**子集**: `normal`

```json
{
  "input": [
    {
      "id": "134dd42a",
      "content": "You are evaluating ACEBench tool-use tasks.\n\nUse the available tool schemas when native function calling is supported.\n\nFor text-only output, return API calls as [ApiName(key1='value1', key2=2)]. Return only the call list and no extra explana ... [TRUNCATED 2553 chars] ... \"}, \"effects\": {\"description\": \"List of audio effects to apply.\", \"type\": \"array\", \"items\": {\"type\": \"string\", \"enum\": [\"reverb\", \"echo\", \"distortion\"]}}}, \"required\": [\"frequency\", \"gain\"]}}}, \"required\": [\"microphone\", \"performanceTime\"]}}]"
    },
    {
      "id": "da9681d9",
      "content": "I have been fascinated recently with total solar eclipses. I am planning my next travel and would like to know when the next total solar eclipse will be visible in Greece, specifically in Athens, over the next five years.",
      "role": "user"
    }
  ],
  "target": "{\"ground_truth\": {\"NightSkyAnalysis_performEclipseAnalysis\": {\"dateRange\": {\"startDate\": \"2023-01-01\", \"endDate\": \"2028-01-01\"}, \"location\": {\"latitude\": 37.9838, \"longitude\": 23.7275}, \"eclipseType\": \"total\"}}, \"mile_stone\": []}",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "NightSkyAnalysis_performEclipseAnalysis",
      "description": "Analyzes the occurrence of solar eclipses, categorizes them into types, and predicts future occurrences based on historical data and celestial mechanics.",
      "parameters": {
        "type": "object",
        "properties": {
          "dateRange": {
            "type": "object",
            "description": "The range of dates for which to analyze solar eclipses.",
            "properties": {
              "startDate": {
                "type": "string",
                "description": "The starting date for the analysis in YYYY-MM-DD format."
              },
              "endDate": {
                "type": "string",
                "description": "The ending date for the analysis in YYYY-MM-DD format."
              }
            },
            "required": [
              "startDate",
              "endDate"
            ]
          },
          "location": {
            "type": "object",
            "description": "Geographical coordinates to focus the eclipse analysis.",
            "properties": {
              "latitude": {
                "type": "number",
                "description": "Latitude of the location."
              },
              "longitude": {
                "type": "number",
                "description": "Longitude of the location."
              }
            },
            "required": [
              "latitude",
              "longitude"
            ]
          },
          "eclipseType": {
            "type": "string",
            "description": "The type of solar eclipse to specifically analyze.",
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
            "type": "object",
            "description": "Details of the microphone used.",
            "properties": {
              "type": {
                "type": "string",
                "description": "Type of the microphone.",
                "enum": [
                  "dynamic",
                  "condenser",
                  "ribbon"
                ]
              },
              "model": {
                "type": "string",
                "description": "Model of the microphone."
              }
            },
            "required": [
              "type",
              "model"
            ]
          },
          "performanceTime": {
            "type": "string",
            "description": "Scheduled time for the performance.",
            "enum": [
              "morning",
              "afternoon",
              "evening",
              "night"
            ]
          },
          "environment": {
            "type": "object",
            "description": "Environmental conditions of the performance area.",
            "properties": {
              "humidity": {
                "type": "integer",
                "description": "Humidity level as a percentage."
              },
              "temperature": {
                "type": "integer",
                "description": "Temperature in Celsius."
              }
            }
          },
          "soundSettings": {
            "type": "array",
            "description": "Specific sound settings to apply.",
            "items": {
              "type": "object",
              "properties": {
                "frequency": {
                  "type": "integer",
                  "description": "Frequency adjustments in Hz."
                },
                "gain": {
                  "type": "integer",
                  "description": "Gain adjustments in dB."
                },
                "effects": {
                  "type": "array",
                  "description": "List of audio effects to apply.",
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
  "metadata": {
    "id": "normal_single_turn_single_function_0",
    "sub_category": "data_normal_single_turn_single_function",
    "question": "user: I have been fascinated recently with total solar eclipses. I am planning my next travel and would like to know when the next total solar eclipse will be visible in Greece, specifically in Athens, over the next five years.\n",
    "time": "",
    "profile": "",
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
    "involved_classes": []
  }
}
```

## 提示模板

*未定义提示模板。*

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
            # subset_list: ['normal', 'special', 'agent']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```