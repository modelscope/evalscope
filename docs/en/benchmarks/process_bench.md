# ProcessBench


## Overview

ProcessBench is a benchmark for evaluating AI models on mathematical reasoning process verification. It tests the ability to identify errors in step-by-step mathematical solutions across various difficulty levels from GSM8K to OmniMath.

## Task Description

- **Task Type**: Mathematical Reasoning Error Detection
- **Input**: Math problem + step-by-step solution (tagged paragraphs)
- **Output**: Index of first error paragraph (or -1 if correct)
- **Domains**: Math reasoning verification, error detection

## Key Features

- Four difficulty subsets:
  - `gsm8k`: Grade school math problems
  - `math`: Competition math problems
  - `olympiadbench`: Olympiad-level problems
  - `omnimath`: Advanced mathematical reasoning
- Tests process supervision and verification abilities
- Requires analyzing step-by-step reasoning for errors

## Evaluation Notes

- Default evaluation uses the **test** split
- Multiple metrics tracked:
  - `error_acc`: Accuracy on detecting error locations
  - `correct_acc`: Accuracy on identifying correct solutions
  - `simple_f1_score`: F1 score balancing both
- Answers should be in \boxed{} format (paragraph index or -1)
- Aggregation method: **F1** score


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `process_bench` |
| **Dataset ID** | [Qwen/ProcessBench](https://modelscope.cn/datasets/Qwen/ProcessBench/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `error_acc`, `correct_acc`, `simple_f1_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `f1` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,400 |
| Prompt Length (Mean) | 2764.83 chars |
| Prompt Length (Min/Max) | 690 / 9005 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `gsm8k` | 400 | 1824.26 | 876 | 4520 |
| `math` | 1,000 | 2297.11 | 690 | 7565 |
| `olympiadbench` | 1,000 | 3166.77 | 1129 | 9005 |
| `omnimath` | 1,000 | 3206.82 | 832 | 8550 |

## Sample Example

**Subset**: `gsm8k`

```json
{
  "input": [
    {
      "id": "aca63163",
      "content": "The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):\n\n[Math Problem]\n\nSue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, t ... [TRUNCATED] ... nce you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes \"not found\").\n\nPlease put your final answer (i.e., the index) in \boxed{}.\n"
    }
  ],
  "target": "1",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "steps": [
      "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
      "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, \\(1/3 \\times 18 = 6\\) flamingos are taken back. So, they have \\(18 - 6 = 12\\) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has \\(12 + 6 = 18\\) pink flamingos and 6 white flamingos.",
      "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has \\(18 + 18 = 36\\) pink flamingos and still 6 white flamingos.",
      "To find the difference, subtract the number of white flamingos from the number of pink flamingos: \\(36 - 6 = 30\\). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is \\(\\boxed{30}\\)."
    ],
    "tagged_response": "<paragraph_0>\nTo find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.\n</paragrap ... [TRUNCATED] ...  subtract the number of white flamingos from the number of pink flamingos: \\(36 - 6 = 30\\). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is \\(\\boxed{30}\\).\n</paragraph_3>",
    "final_answer_correct": false
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in oxed{{}}.

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets process_bench \
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
    datasets=['process_bench'],
    dataset_args={
        'process_bench': {
            # subset_list: ['gsm8k', 'math', 'olympiadbench']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


