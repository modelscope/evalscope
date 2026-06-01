# ACEBench


## Overview

ACEBench is a tool-use benchmark for evaluating whether large language models can select APIs, fill
arguments, handle abnormal requests, and complete realistic agent tasks.

## Task Description

- **Task Type**: Function calling and agentic tool use
- **Input**: Conversation history, API specifications, optional time/profile context, and agent task context
- **Output**: Function calls or diagnostic text for special cases
- **Subsets**: normal, special, and agent

## Evaluation Notes

- The adapter passes ACEBench API specifications as EvalScope tools and also includes concise text
  instructions for text-only models.
- Normal samples are scored by matching function names and arguments.
- Special samples are scored against ACEBench's diagnostic text contract.
- Agent samples report `process_acc` against ACEBench milestones. If a model returns a final-state JSON object,
  `end_state_acc` is also reported and used as `acc`; otherwise `acc` follows `process_acc`.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `acebench` |
| **Dataset ID** | [evalscope/acebench](https://modelscope.cn/datasets/evalscope/acebench/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **Metrics** | `acc`, `process_acc`, `end_state_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `normal` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,023 |
| Prompt Length (Mean) | 4676.74 chars |
| Prompt Length (Min/Max) | 1007 / 10555 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `normal` | 823 | 4859.68 | 1070 | 10555 |
| `special` | 150 | 3449.69 | 1007 | 8692 |
| `agent` | 50 | 5346.72 | 4454 | 5656 |

## Sample Example

**Subset**: `normal`

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

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets acebench \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

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
            # subset_list: ['normal', 'special', 'agent']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


