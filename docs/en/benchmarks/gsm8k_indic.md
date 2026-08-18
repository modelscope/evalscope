# GSM8K-Indic


## Overview

GSM8K-Indic translates the GSM8K grade-school math word problems into 10 Indic languages, each
available in native script and a romanized (Latin transliteration) variant, plus the original English.

## Task Description

- **Task Type**: Multilingual Mathematical Word Problem Solving
- **Input**: Grade-school math word problem in one of 21 language/script variants
- **Output**: Numerical answer derived through step-by-step reasoning
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil,
  Telugu — each Indic language in both native script and a `_roman` transliterated variant

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate specific languages/scripts (e.g., `['hi', 'hi_roman']`)
- Gold answers are the original English reasoning chain's final numeric value; only the question is
  translated


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gsm8k_indic` |
| **Dataset ID** | [sarvamai/gsm8k-indic](https://modelscope.cn/datasets/sarvamai/gsm8k-indic/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `MultiLingual`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 27,670 |
| Prompt Length (Mean) | 335.91 chars |
| Prompt Length (Min/Max) | 135 / 1045 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en` | 1,319 | 310.87 | 144 | 919 |
| `bn` | 1,319 | 306.06 | 136 | 790 |
| `gu` | 1,319 | 303.06 | 135 | 770 |
| `hi` | 1,319 | 311.1 | 147 | 761 |
| `kn` | 1,319 | 338.11 | 145 | 831 |
| `ml` | 1,319 | 345.21 | 156 | 907 |
| `mr` | 1,319 | 315.01 | 146 | 785 |
| `or` | 1,319 | 313.28 | 141 | 815 |
| `pa` | 1,319 | 312.77 | 152 | 800 |
| `ta` | 1,319 | 367.12 | 161 | 972 |
| `te` | 1,319 | 330.78 | 142 | 877 |
| `bn_roman` | 1,319 | 330.69 | 153 | 816 |
| `gu_roman` | 1,318 | 335.73 | 148 | 885 |
| `hi_roman` | 1,319 | 340.77 | 154 | 869 |
| `kn_roman` | 1,316 | 368.49 | 149 | 959 |
| `ml_roman` | 1,319 | 363.39 | 159 | 937 |
| `mr_roman` | 1,310 | 339.56 | 156 | 848 |
| `or_roman` | 1,319 | 339.92 | 158 | 930 |
| `pa_roman` | 1,319 | 334.74 | 155 | 847 |
| `ta_roman` | 1,303 | 387.6 | 163 | 1045 |
| `te_roman` | 1,319 | 360.46 | 145 | 1025 |

## Sample Example

**Subset**: `en`

```json
{
  "input": [
    {
      "id": "92205129",
      "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gsm8k_indic \
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
    datasets=['gsm8k_indic'],
    dataset_args={
        'gsm8k_indic': {
            # subset_list: ['en', 'bn', 'gu']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
