# ACEBench


## Overview

ACEBench evaluates whether large language models can use tools in realistic settings: picking the
right API, filling its arguments, pushing back on requests that cannot be satisfied, and driving
multi-step agent tasks against a simulated environment. Data is split into three families -
`normal` (ordinary tool use), `special` (incomplete, incorrect or out-of-scope requests) and
`agent` (multi-step and multi-turn interaction) - reported over 17 fine-grained categories.

## Task Description

- **Task Type**: Function calling and agentic tool use
- **Input**: Conversation history, API specifications, and optional time or character-profile context
- **Output**: A `[ApiName(key='value')]` call list, a diagnostic sentence, or a full agent trajectory
- **Domain**: 8 domains and 68 sub-domains including technology, finance, health and society

## Key Features

- 1023 English and 1017 Chinese samples, selectable through `extra_params.language`.
- Uses the official ACEBench prompts and the official `[ApiName(...)]` output contract, so an
  output that cannot be decoded scores zero instead of being rescued by lenient parsing.
- `normal_multi_turn_*` categories are scored per dialogue: every step must be correct for the
  dialogue to count, matching the official turn-level aggregation.
- `agent` categories run a real rollout against ACEBench's simulated phone, food-delivery and
  travel APIs, and are graded on the resulting environment state.

## Evaluation Notes

- `acc` is the primary metric. For `normal` and `special` it is answer accuracy; for `agent` it is
  end-state accuracy. `process_acc` additionally reports milestone progress for `agent` samples and
  per-step progress for `normal_multi_turn_*` samples.
- The report adds the official groupings (ATOM, SINGLE_TURN, MULTI_TURN, NORMAL, SPECIAL, AGENT)
  and an OVERALL score weighted `normal` 0.578 / `special` 0.2676 / `agent` 0.1545. Weights are
  renormalized over the groups actually evaluated, so a partial run stays interpretable.
- `agent_multi_turn` additionally needs a user simulator; set `extra_params.user_model` to the model
  that should play the user (the official runner uses `gpt-4o`). Without it those rollouts fail and
  score zero, so configure it before reading an OVERALL number.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `acebench` |
| **Dataset ID** | [evalscope/acebench](https://modelscope.cn/datasets/evalscope/acebench/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling`, `MultiTurn` |
| **Metrics** | `acc`, `process_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `normal` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,023 |
| Prompt Length (Mean) | 6032.98 chars |
| Prompt Length (Min/Max) | 2295 / 11835 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
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

## Sample Example

**Subset**: `normal_single_turn_single_function`

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

*Note: Some content was truncated for display.*

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `en` | Dataset language to evaluate, either `en` or `zh`. |
| `user_model` | `str` | `` | Model that plays the user in `agent_multi_turn` rollouts, e.g. `gpt-4o`. Those rollouts fail and score zero when unset. |
| `user_model_api_url` | `str` | `` | Base URL for `user_model`. Defaults to `MODELSCOPE_API_BASE`. |
| `user_model_api_key` | `str` | `` | API key for `user_model`. Defaults to `MODELSCOPE_SDK_TOKEN`. |
| `max_dialog_turns` | `int` | `40` | Maximum number of agent rollout steps. |

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
            # subset_list: ['normal_single_turn_single_function', 'normal_single_turn_parallel_function', 'normal_multi_turn_user_adjust']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
