# Global-MMLU


## Overview

Global-MMLU evaluates multilingual knowledge and performance on culturally sensitive and agnostic questions.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 42 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: The 57 MMLU subjects

## Key Features

- 589,764 test questions across 42 languages. Each language has 2,850 culturally labeled questions.
- Keeps the 39 rows with a missing option, including 12 where the correct choice is blank.

## Evaluation Notes

- Uses five same-subject examples from the language's `dev` split where available.
- Scores generated answer letters by exact match. The upstream harness scores choice likelihoods and its Lite tasks
  select examples by subject category.
- Reports overall, culturally sensitive, and culturally agnostic accuracy per language. Bayes@N reports overall only.
- Licensed under [Apache 2.0](https://opensource.org/license/apache-2-0).


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `global_mmlu` |
| **Dataset ID** | [CohereLabs/Global-MMLU](https://modelscope.cn/datasets/CohereLabs/Global-MMLU/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2412.03304) |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `dev` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 589,764 |
| Prompt Length (Mean) | 3048.64 chars |
| Prompt Length (Min/Max) | 558 / 16835 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `am` | 14,042 | 2143.94 | 749 | 8937 |
| `ar` | 14,042 | 2718.89 | 830 | 11584 |
| `bn` | 14,042 | 2985.86 | 850 | 13433 |
| `cs` | 14,042 | 2955.66 | 863 | 13127 |
| `de` | 14,042 | 3570.76 | 1007 | 16685 |
| `el` | 14,042 | 3446.37 | 1003 | 15859 |
| `en` | 14,042 | 3178.85 | 949 | 14594 |
| `es` | 14,042 | 3469.11 | 977 | 15556 |
| `fa` | 14,042 | 2772.79 | 821 | 12018 |
| `fil` | 14,042 | 3641.24 | 997 | 16835 |
| `fr` | 14,042 | 3554.29 | 1015 | 15946 |
| `ha` | 14,042 | 3183.13 | 890 | 14093 |
| `he` | 14,042 | 2411.77 | 813 | 10445 |
| `hi` | 14,042 | 3087.54 | 900 | 13884 |
| `id` | 14,042 | 3447.94 | 938 | 15651 |
| `ig` | 14,042 | 2998.41 | 896 | 13562 |
| `it` | 14,042 | 3482.2 | 984 | 15316 |
| `ja` | 14,042 | 1571.26 | 627 | 6229 |
| `ko` | 14,042 | 1733.23 | 661 | 7096 |
| `ky` | 14,042 | 3121.07 | 936 | 14341 |
| `lt` | 14,042 | 3053.71 | 921 | 13664 |
| `mg` | 14,042 | 3567.95 | 1042 | 16250 |
| `ms` | 14,042 | 3345.47 | 947 | 15393 |
| `ne` | 14,042 | 2893.76 | 861 | 13033 |
| `nl` | 14,042 | 3488.92 | 993 | 16074 |
| `ny` | 14,042 | 3307.43 | 1012 | 15235 |
| `pl` | 14,042 | 3276.01 | 909 | 14503 |
| `pt` | 14,042 | 3295.65 | 970 | 14620 |
| `ro` | 14,042 | 3350.91 | 932 | 14854 |
| `ru` | 14,042 | 3319.02 | 920 | 14635 |
| `si` | 14,042 | 2958.06 | 881 | 13125 |
| `sn` | 14,042 | 3217.21 | 962 | 14630 |
| `so` | 14,042 | 3444.59 | 944 | 15603 |
| `sr` | 14,042 | 2976.99 | 901 | 13118 |
| `sv` | 14,042 | 3116.67 | 924 | 14506 |
| `sw` | 14,042 | 3214.23 | 941 | 14343 |
| `te` | 14,042 | 3044.29 | 900 | 13672 |
| `tr` | 14,042 | 3094.77 | 904 | 14003 |
| `uk` | 14,042 | 3112.33 | 908 | 13695 |
| `vi` | 14,042 | 3253.18 | 977 | 15261 |
| `yo` | 14,042 | 3002.93 | 938 | 13722 |
| `zh` | 14,042 | 1234.39 | 558 | 4397 |

## Sample Example

**Subset**: `am`

```json
{
  "input": [
    {
      "id": "767e9bb5",
      "content": "Here are some examples of how to answer similar questions:\n\nZ_3[x]/(x^2 + c) መስክ እንደሆነ ሁሉ በZ_3 ውስጥ ያግኙ።\nA) 0\nB) 1\nC) 2\nD) 3\nANSWER: B\n\nመግለጫ 1 | aH የአንድ ፋክተር ቡድን ኤለመንት ከሆነ |aH| ይከፋፍላል |a|. መግለጫ 2 | H እና K የጂ ንዑስ ቡድን ከሆኑ HK የጂ ንዑስ ቡድን ነው።\nA) እው ... [TRUNCATED 469 chars] ... le choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nየተሰጠውን የመስክ ማራዘሚያ Q(sqrt(2)፣ sqrt(3)፣ sqrt(18)) ከQ በላይ አግኝ።\n\nA) 0\nB) 4\nC) 2\nD) 6"
    }
  ],
  "choices": [
    "0",
    "4",
    "2",
    "6"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "sample_id": "abstract_algebra/test/0",
    "subject": "abstract_algebra",
    "subject_category": "STEM",
    "cultural_sensitivity_label": "-",
    "language": "Amharic"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets global_mmlu \
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
    datasets=['global_mmlu'],
    dataset_args={
        'global_mmlu': {
            # subset_list: ['am', 'ar', 'bn']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
