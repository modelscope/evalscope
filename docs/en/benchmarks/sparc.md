# SPaRC


## Overview

SPaRC (Spatial Pathfinding Reasoning Challenge) tests step-by-step spatial reasoning with 2D grid
puzzles based on the game *The Witness*. The model must draw one line from a start node to an end
node that obeys every rule on the grid.

## Task Description

- **Task Type**: Spatial pathfinding under rule constraints
- **Input**: A text grid (start and end nodes, gaps, dots, colored squares, stars, triangles,
  polyominoes) and the rules of the game
- **Output**: Step-by-step reasoning, then the marker `####` and the path as a list of `(x, y)`
  node coordinates
- **Domain**: Spatial reasoning, constraint satisfaction

## Key Features

- 500 test puzzles in five difficulty levels (Level 1 to Level 5: 86 / 118 / 121 / 86 / 89 puzzles);
  the paper calls Level 5 "hard"
- Every valid solution of each puzzle is stored in the dataset, so scoring needs no judge model
- In the paper, humans solve 98.0% of the test puzzles (94.5% of Level 5), and the best model,
  o4-mini, solves 15.8% (1.1% of Level 5)
- The prompt is the paper's single-shot prompt, with one worked example inside the prompt

## Evaluation Notes

- Subsets are the difficulty levels; `subset_list: ['Level 5']` evaluates only the hard puzzles.
  `limit` applies per level, so `limit=4` gives 20 puzzles (4 per level)
- `accuracy` (primary): the path after the last `####` must exactly match one of the puzzle's
  stored solutions
- `invalid_path_rate` (lower is better): share of answers that are not a valid line from start to
  end (wrong start or end, a gap between steps, a revisited node, or a step on a rule cell). This
  is the "Invalid Path" column of Table 7 in the paper
- The prompt builder and the validator are copies of the reference code in the `sparc-puzzle`
  package; they give the same prompts and the same scores on all 500 test puzzles
- The dataset is `evalscope/SPaRC` on ModelScope (config `all`), with the revision pinned for
  reproducibility
- Reasoning models often write more than 10k tokens per puzzle, so set a large `max_tokens`
- [Paper](https://aclanthology.org/2025.emnlp-main.526/) | [Code](https://github.com/lkaesberg/SPaRC) |
  [Dataset](https://www.modelscope.cn/datasets/evalscope/SPaRC) | [Website](https://sparc.gipplab.org/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `sparc` |
| **Dataset ID** | [evalscope/SPaRC](https://modelscope.cn/datasets/evalscope/SPaRC/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2505.16686) |
| **Tags** | `Reasoning` |
| **Metrics** | `accuracy`, `invalid_path_rate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 500 |
| Prompt Length (Mean) | 7753.07 chars |
| Prompt Length (Min/Max) | 7478 / 8589 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Level 1` | 86 | 7565.24 | 7478 | 7746 |
| `Level 2` | 118 | 7668.82 | 7484 | 7963 |
| `Level 3` | 121 | 7715.45 | 7486 | 8018 |
| `Level 4` | 86 | 7848.93 | 7594 | 8184 |
| `Level 5` | 89 | 8004.79 | 7610 | 8589 |

## Sample Example

**Subset**: `Level 1`

```json
{
  "input": [
    {
      "id": "a7ac7618",
      "content": "You are an expert at solving puzzles games."
    },
    {
      "id": "468c49a5",
      "content": "\n    ## Objective\n    You are a specialized AI proficient in spatial reasoning and solving puzzles from the game 'The Witness'. Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provide ... [TRUNCATED 7215 chars] ...  marker string `####` followed immediately by the solution path as a list of node coordinates `(x, y)`. Include all intermediate nodes from start to end.\n\n    **Example Solution Path Format:**\n    ####\n    (0,0), (1,0), (2,0), (2,1), ...\n    "
    }
  ],
  "target": "(5,0), (6,0), (6,1), (6,2), (5,2), (4,2), (4,3), (4,4), (5,4), (6,4), (6,5), (6,6), (5,6), (4,6), (3,6), (2,6), (1,6), (0,6), (0,5), (0,4), (1,4), (2,4), (2,3), (2,2), (1,2), (0,2), (0,1), (0,0), (1,0), (2,0), (3,0), (4,0)",
  "id": 0,
  "group_id": 0,
  "subset_key": "Level 1",
  "metadata": {
    "difficulty_level": 1,
    "difficulty_score": 0.842447049854198,
    "solution_count": 3,
    "puzzle": {
      "id": "3fe8a85b768ae5b7",
      "grid_size": {
        "height": 3,
        "width": 4
      },
      "puzzle_array": [
        [
          "+",
          ".",
          "+",
          "+",
          "E",
          "S",
          "+",
          "+",
          "+"
        ],
        [
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+",
          "P-B-1824",
          "+"
        ],
        [
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+"
        ],
        [
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "G"
        ],
        [
          ".",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+",
          "+"
        ],
        [
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+",
          "N",
          "+"
        ],
        [
          "+",
          "+",
          "+",
          "+",
          "+",
          ".",
          "+",
          "+",
          "+"
        ]
      ],
      "polyshapes": "{\"1824\":[[0,0,1,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]]}",
      "solutions": [
        {
          "index": 0,
          "path": [
            {
              "x": 5,
              "y": 0
            },
            {
              "x": 6,
              "y": 0
            },
            {
              "x": 6,
              "y": 1
            },
            {
              "x": 6,
              "y": 2
            },
            {
              "x": 5,
              "y": 2
            },
            {
              "x": 4,
              "y": 2
            },
            {
              "x": 4,
              "y": 3
            },
            {
              "x": 4,
              "y": 4
            },
            {
              "x": 5,
              "y": 4
            },
            {
              "x": 6,
              "y": 4
            },
            "... [TRUNCATED 22 more items] ..."
          ],
          "pathLength": 32
        },
        {
          "index": 1,
          "path": [
            {
              "x": 5,
              "y": 0
            },
            {
              "x": 6,
              "y": 0
            },
            {
              "x": 6,
              "y": 1
            },
            {
              "x": 6,
              "y": 2
            },
            {
              "x": 5,
              "y": 2
            },
            {
              "x": 4,
              "y": 2
            },
            {
              "x": 4,
              "y": 3
            },
            {
              "x": 4,
              "y": 4
            },
            {
              "x": 5,
              "y": 4
            },
            {
              "x": 6,
              "y": 4
            },
            "... [TRUNCATED 18 more items] ..."
          ],
          "pathLength": 28
        },
        {
          "index": 2,
          "path": [
            {
              "x": 5,
              "y": 0
            },
            {
              "x": 6,
              "y": 0
            },
            {
              "x": 6,
              "y": 1
            },
            {
              "x": 6,
              "y": 2
            },
            {
              "x": 5,
              "y": 2
            },
            {
              "x": 4,
              "y": 2
            },
            {
              "x": 4,
              "y": 3
            },
            {
              "x": 4,
              "y": 4
            },
            {
              "x": 5,
              "y": 4
            },
            {
              "x": 6,
              "y": 4
            },
            "... [TRUNCATED 18 more items] ..."
          ],
          "pathLength": 28
        }
      ]
    }
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**System Prompt:**
```text
You are an expert at solving puzzles games.
```

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets sparc \
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
    datasets=['sparc'],
    dataset_args={
        'sparc': {
            # subset_list: ['Level 1', 'Level 2', 'Level 3']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
