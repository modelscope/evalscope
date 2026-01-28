# BFCL-v3


## Overview

BFCL (Berkeley Function Calling Leaderboard) v3 is the first comprehensive and executable function call evaluation benchmark for assessing LLMs' ability to invoke functions. It evaluates various forms of function calls, diverse scenarios, and executability.

## Task Description

- **Task Type**: Function Calling / Tool Use Evaluation
- **Input**: User query with available function definitions
- **Output**: Correct function call with parameters
- **Categories**: AST_NON_LIVE, AST_LIVE, RELEVANCE, MULTI_TURN

## Key Features

- Comprehensive function call evaluation
- Tests simple, multiple, and parallel function calls
- Multi-language support (Python, Java, JavaScript)
- Relevance detection (irrelevance handling)
- Multi-turn conversation function calling
- Live API endpoints for execution testing

## Evaluation Notes

- Requires `pip install bfcl-eval==2025.10.27.1` before evaluation
- Set `is_fc_model=True` for models with native function calling support
- `underscore_to_dot` parameter controls function name formatting
- See the [usage documentation](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html) for detailed setup
- Results broken down by category (AST, Relevance, Multi-turn)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bfcl_v3` |
| **Dataset ID** | [AI-ModelScope/bfcl_v3](https://modelscope.cn/datasets/AI-ModelScope/bfcl_v3/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,441 |
| Prompt Length (Mean) | 291.25 chars |
| Prompt Length (Min/Max) | 36 / 10805 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
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

## Sample Example

**Subset**: `simple`

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

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `underscore_to_dot` | `bool` | `True` | Convert underscores to dots in function names for evaluation. |
| `is_fc_model` | `bool` | `True` | Indicates the evaluated model natively supports function calling. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bfcl_v3 \
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
    datasets=['bfcl_v3'],
    dataset_args={
        'bfcl_v3': {
            # subset_list: ['simple', 'multiple', 'parallel']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


