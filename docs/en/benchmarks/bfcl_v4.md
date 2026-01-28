# BFCL-v4


## Overview

BFCL-v4 (Berkeley Function-Calling Leaderboard V4) is a comprehensive benchmark for evaluating agentic function-calling capabilities of LLMs. It tests web search, memory operations, and format sensitivity as building blocks for agentic applications.

## Task Description

- **Task Type**: Agentic Function Calling Evaluation
- **Input**: Scenarios requiring function calls for web search, memory, etc.
- **Output**: Correct function calls with proper arguments
- **Capabilities**: Web search, memory read/write, format handling

## Key Features

- Holistic agentic evaluation framework
- Tests building blocks for LLM agents (search, memory, formatting)
- Multiple scoring categories for detailed analysis
- Supports both FC models and non-FC models
- Optional web search with SerpAPI integration

## Evaluation Notes

- **Installation Required**: `pip install bfcl-eval==2025.10.27.1`
- Primary metric: **Accuracy** per category
- Configure `is_fc_model` based on model capabilities
- Optional: Set `SERPAPI_API_KEY` for web search tasks
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v4.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bfcl_v4` |
| **Dataset ID** | [berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| **Paper** | N/A |
| **Tags** | `Agent`, `FunctionCalling` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 5,106 |
| Prompt Length (Mean) | 350.4 chars |
| Prompt Length (Min/Max) | 36 / 10805 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
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

## Sample Example

**Subset**: `irrelevance`

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

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `underscore_to_dot` | `bool` | `True` | Convert underscores to dots in function names for evaluation. |
| `is_fc_model` | `bool` | `True` | Indicates the evaluated model natively supports function calling. |
| `SERPAPI_API_KEY` | `str | null` | `None` | SerpAPI key enabling web-search capability in BFCL V4. Null disables web search. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bfcl_v4 \
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
    datasets=['bfcl_v4'],
    dataset_args={
        'bfcl_v4': {
            # subset_list: ['irrelevance', 'live_irrelevance', 'live_multiple']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


