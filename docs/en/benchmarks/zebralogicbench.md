# ZebraLogicBench


## Overview

ZebraLogicBench is a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs). It tests systematic logical reasoning abilities.

## Task Description

- **Task Type**: Logic Grid Puzzle Solving
- **Input**: Logic puzzle with houses, attributes, and clues
- **Output**: JSON solution with reasoning explanation
- **Domains**: Constraint satisfaction, logical deduction

## Key Features

- Puzzles derived from constraint satisfaction problems
- Requires systematic step-by-step logical reasoning
- Varying difficulty levels (Easy/Hard) and sizes (Small/Medium/Large/XL)
- Tests ability to process multiple interdependent clues
- Solutions must be valid JSON format

## Evaluation Notes

- Default evaluation uses the **test** split with **zero-shot**
- Multiple metrics tracked:
  - `puzzle_acc`: Correctly solved complete puzzles
  - `cell_acc`: Correctly identified individual cells
  - Difficulty-based: `easy_puzzle_acc`, `hard_puzzle_acc`
  - Size-based: `small`, `medium`, `large`, `xl_puzzle_acc`
  - `avg_reason_lens`: Average reasoning length
- Output must include reasoning and solution in JSON format


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `zebralogicbench` |
| **Dataset ID** | [allenai/ZebraLogicBench-private](https://modelscope.cn/datasets/allenai/ZebraLogicBench-private/summary) |
| **Paper** | N/A |
| **Tags** | `Reasoning` |
| **Metrics** | `puzzle_acc`, `cell_acc`, `easy_puzzle_acc`, `hard_puzzle_acc`, `small_puzzle_acc`, `medium_puzzle_acc`, `large_puzzle_acc`, `xl_puzzle_acc`, `avg_reason_lens`, `no_answer_num` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 3262.38 chars |
| Prompt Length (Min/Max) | 2011 / 5658 chars |

## Sample Example

**Subset**: `grid_mode`

```json
{
  "input": [
    {
      "id": "e6c901c7",
      "content": "# Example Puzzle\n\nThere are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:\n - Each perso ... [TRUNCATED] ... Animal\": \"___\"\n        },\n        \"House 5\": {\n            \"Name\": \"___\",\n            \"Nationality\": \"___\",\n            \"BookGenre\": \"___\",\n            \"Food\": \"___\",\n            \"Color\": \"___\",\n            \"Animal\": \"___\"\n        }\n    }\n}\n\n"
    }
  ],
  "target": "{\"header\": [\"House\", \"Name\", \"Nationality\", \"BookGenre\", \"Food\", \"Color\", \"Animal\"], \"rows\": [[\"1\", \"Bob\", \"german\", \"mystery\", \"grilled cheese\", \"yellow\", \"dog\"], [\"2\", \"Eric\", \"norwegian\", \"fantasy\", \"stew\", \"blue\", \"fish\"], [\"3\", \"Peter\", \"dane\", \"science fiction\", \"spaghetti\", \"green\", \"cat\"], [\"4\", \"Arnold\", \"swede\", \"biography\", \"stir fry\", \"red\", \"bird\"], [\"5\", \"Alice\", \"brit\", \"romance\", \"pizza\", \"white\", \"horse\"]]}",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "created_at": "2024-07-03T21:21:29.209499"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {{
        "House 1": {{
            "Name": "Arnold",
            "Drink": "tea"
        }},
        "House 2": {{
            "Name": "Peter",
            "Drink": "water"
        }},
        "House 3": {{
            "Name": "Eric",
            "Drink": "milk"
        }}
    }}
}}

# Puzzle to Solve

{question}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}


```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets zebralogicbench \
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
    datasets=['zebralogicbench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


