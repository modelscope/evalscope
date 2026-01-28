# ChartQA


## Overview

ChartQA is a benchmark designed to evaluate question-answering capabilities over charts and data visualizations. It tests both visual reasoning and logical understanding of various chart types including bar charts, line graphs, and pie charts.

## Task Description

- **Task Type**: Chart Question Answering
- **Input**: Chart image + natural language question
- **Output**: Single word or numerical answer
- **Domains**: Data visualization, visual reasoning, numerical reasoning

## Key Features

- Covers diverse chart types (bar, line, pie, scatter plots)
- Includes both human-written and augmented test questions
- Requires understanding of chart structure and data relationships
- Tests both visual extraction and logical reasoning abilities
- Questions range from simple data lookup to complex reasoning

## Evaluation Notes

- Default evaluation uses the **test** split with two subsets:
  - `human_test`: Human-written questions
  - `augmented_test`: Automatically generated questions
- Primary metric: **Relaxed Accuracy** (allows minor variations in answers)
- Answers should be in format "ANSWER: [ANSWER]"
- Numerical answers may have tolerance for rounding differences


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `chartqa` |
| **Dataset ID** | [lmms-lab/ChartQA](https://modelscope.cn/datasets/lmms-lab/ChartQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `relaxed_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,500 |
| Prompt Length (Mean) | 224.33 chars |
| Prompt Length (Min/Max) | 178 / 352 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `human_test` | 1,250 | 220.17 | 178 | 352 |
| `augmented_test` | 1,250 | 228.49 | 186 | 293 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,500 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 184x326 - 800x1796 |
| Formats | png |


## Sample Example

**Subset**: `human_test`

```json
{
  "input": [
    {
      "id": "c75439e0",
      "content": [
        {
          "text": "\nHow many food item is shown in the bar graph?\n\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the a single word answer or number to the problem.\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~42.9KB]"
        }
      ]
    }
  ],
  "target": "14",
  "id": 0,
  "group_id": 0,
  "subset_key": "human_test"
}
```

## Prompt Template

**Prompt Template:**
```text

{question}

The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the a single word answer or number to the problem.

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets chartqa \
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
    datasets=['chartqa'],
    dataset_args={
        'chartqa': {
            # subset_list: ['human_test', 'augmented_test']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


