# BBH


## Overview

BBH (BIG-Bench Hard) is a subset of 23 challenging tasks from the BIG-Bench benchmark that are specifically selected because language models initially struggled with them. These tasks require complex reasoning abilities that benefit from Chain-of-Thought (CoT) prompting.

## Task Description

- **Task Type**: Mixed (Multiple-Choice and Free-Form)
- **Input**: Task-specific questions requiring reasoning
- **Output**: Answers in specified format
- **Subsets**: 27 reasoning tasks divided into multiple-choice (17) and free-form (10)

## Key Features

- 27 challenging reasoning tasks from BIG-Bench
- Multiple-choice tasks: temporal sequences, disambiguation, logical deduction, etc.
- Free-form tasks: arithmetic, navigation, boolean expressions, etc.
- Each task comes with curated Chain-of-Thought examples
- Designed to test advanced reasoning capabilities

## Evaluation Notes

- Default configuration uses **3-shot** with CoT prompting (recommended)
- CoT prompts are pre-defined for each subset in `cot_prompts/` directory
- Answers should follow the format: "So the answer is [ANSWER]"
- Setting `few_shot_num=0` disables few-shot examples
- Multiple-choice answers are normalized to single letters (A, B, C, etc.)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bbh` |
| **Dataset ID** | [evalscope/bbh](https://modelscope.cn/datasets/evalscope/bbh/summary) |
| **Paper** | N/A |
| **Tags** | `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 3-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 6,511 |
| Prompt Length (Mean) | 3307.29 chars |
| Prompt Length (Min/Max) | 1060 / 7885 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `temporal_sequences` | 250 | 3746.18 | 3646 | 3876 |
| `disambiguation_qa` | 250 | 4047.48 | 3993 | 4099 |
| `date_understanding` | 250 | 1550.66 | 1491 | 1641 |
| `tracking_shuffled_objects_three_objects` | 250 | 3257.42 | 3195 | 3316 |
| `penguins_in_a_table` | 146 | 3030.88 | 2922 | 3201 |
| `geometric_shapes` | 250 | 5270.24 | 5201 | 5384 |
| `snarks` | 178 | 3493.68 | 3339 | 3693 |
| `ruin_names` | 250 | 3832.01 | 3781 | 3948 |
| `tracking_shuffled_objects_seven_objects` | 250 | 3598.1 | 3506 | 3682 |
| `tracking_shuffled_objects_five_objects` | 250 | 3419.36 | 3338 | 3489 |
| `logical_deduction_three_objects` | 250 | 3093.32 | 3014 | 3165 |
| `hyperbaton` | 250 | 3433.3 | 3386 | 3486 |
| `logical_deduction_five_objects` | 250 | 3264.38 | 3118 | 3379 |
| `logical_deduction_seven_objects` | 250 | 3434.09 | 3217 | 3633 |
| `movie_recommendation` | 250 | 2489.85 | 2436 | 2613 |
| `salient_translation_error_detection` | 250 | 7401.64 | 7223 | 7885 |
| `reasoning_about_colored_objects` | 250 | 2818.32 | 2572 | 3102 |
| `multistep_arithmetic_two` | 250 | 2596.98 | 2594 | 2600 |
| `navigate` | 250 | 2508.7 | 2452 | 2626 |
| `dyck_languages` | 250 | 2723.8 | 2680 | 2874 |
| `word_sorting` | 250 | 2481.34 | 2397 | 2569 |
| `sports_understanding` | 250 | 1077.42 | 1060 | 1122 |
| `boolean_expressions` | 250 | 1991.7 | 1980 | 1998 |
| `object_counting` | 250 | 1706.66 | 1647 | 1787 |
| `formal_fallacies` | 250 | 5185.5 | 4918 | 5514 |
| `causal_judgement` | 187 | 4877.42 | 4194 | 6311 |
| `web_of_lies` | 250 | 3300.84 | 3267 | 3340 |

## Sample Example

**Subset**: `temporal_sequences`

```json
{
  "input": [
    {
      "id": "7d1767c8",
      "content": "Task description: Answer questions about which times certain events could have occurred.\n\nQ: Today, Emily went to the museum. Between what times could they have gone?\nWe know that:\nEmily woke up at 1pm.\nElizabeth saw Emily reading at the libr ... [TRUNCATED] ... \nOptions:\n(A) 6pm to 9pm\n(B) 7am to 11am\n(C) 1pm to 2pm\n(D) 2pm to 6pm\nA: Let's think step by step. Put your final answer in the format of \"So the answer is [ANSWER]\" (without quotes and markdown) where [ANSWER] is the answer to the problem.\n"
    }
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "temporal_sequences",
  "metadata": {
    "task_type": "multiple_choice"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Q: {question}
A: Let's think step by step. Put your final answer in the format of "So the answer is [ANSWER]" (without quotes and markdown) where [ANSWER] is the answer to the problem.

```

<details>
<summary>Few-shot Template</summary>

```text
{fewshot}

Q: {question}
A: Let's think step by step. Put your final answer in the format of "So the answer is [ANSWER]" (without quotes and markdown) where [ANSWER] is the answer to the problem.

```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bbh \
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
    datasets=['bbh'],
    dataset_args={
        'bbh': {
            # subset_list: ['temporal_sequences', 'disambiguation_qa', 'date_understanding']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


