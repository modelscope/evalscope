# Global-MMLU-Lite


## Overview

Global-MMLU-Lite balances culturally sensitive and agnostic questions across 23 human-reviewed languages.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 23 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: 43 of the 57 MMLU subjects

## Key Features

- Version 3.0 evaluates 9,199 questions after skipping one blank Burmese question. The paper covers the first 15 languages.
- Welsh uses zero-shot because its release has no `dev` split.

## Evaluation Notes

- Uses five same-subject examples from the language's `dev` split where available.
- Scores generated answer letters by exact match. The upstream harness scores choice likelihoods and its Lite tasks
  select examples by subject category.
- Reports overall, culturally sensitive, and culturally agnostic accuracy per language. Bayes@N reports overall only.
- Licensed under [Apache 2.0](https://opensource.org/license/apache-2-0).


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `global_mmlu_lite` |
| **Dataset ID** | [CohereLabs/Global-MMLU-Lite](https://modelscope.cn/datasets/CohereLabs/Global-MMLU-Lite/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2412.03304) |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `dev` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 9,199 |
| Prompt Length (Mean) | 2616.82 chars |
| Prompt Length (Min/Max) | 252 / 16179 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `ar` | 400 | 2469.9 | 995 | 11150 |
| `bn` | 400 | 2687.55 | 987 | 12825 |
| `cs` | 400 | 2680.58 | 958 | 12785 |
| `cy` | 400 | 615.85 | 252 | 2783 |
| `de` | 400 | 3215.62 | 1037 | 16179 |
| `en` | 400 | 2873.17 | 984 | 14330 |
| `es` | 400 | 3127.28 | 1019 | 15071 |
| `fr` | 400 | 3207.97 | 1037 | 15429 |
| `hi` | 400 | 2789.74 | 966 | 13386 |
| `hu` | 400 | 2978.19 | 996 | 14592 |
| `id` | 400 | 3073.08 | 1030 | 15137 |
| `it` | 400 | 3115.24 | 1024 | 15079 |
| `ja` | 400 | 1444.81 | 718 | 6013 |
| `ko` | 400 | 1571.24 | 709 | 6645 |
| `my` | 399 | 3147.32 | 1018 | 15746 |
| `or` | 400 | 2726.38 | 995 | 13442 |
| `pt` | 400 | 2974.53 | 994 | 14253 |
| `sk` | 400 | 2772.8 | 977 | 13133 |
| `sq` | 400 | 3030.12 | 1033 | 14570 |
| `sw` | 400 | 2882.58 | 1046 | 14042 |
| `tg` | 400 | 2955.62 | 981 | 13788 |
| `yo` | 400 | 2696.96 | 984 | 13569 |
| `zh` | 400 | 1151.58 | 624 | 4255 |

## Sample Example

**Subset**: `ar`

```json
{
  "input": [
    {
      "id": "87416dc0",
      "content": "Here are some examples of how to answer similar questions:\n\nأنت تدفع شاحنة على طول الطريق. هل سيكون من الأسهل تسريع هذه الشاحنة على المريخ؟ لماذا؟ (افترض عدم وجود احتكاك)\nA) سيكون الأمر أصعب لأن الشاحنة أثقل وزنًا على المريخ.\nB) سيكون الأمر أ ... [TRUNCATED 1714 chars] ...  following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nعند السفر شمالًا من الولايات المتحدة إلى كندا، سترى نجم الشمال (بولاريس) _________.\n\nA) أكثر سطوعًا\nB) باهت\nC) أعلى في السماء\nD) أكثر انخفاضًا في السماء"
    }
  ],
  "choices": [
    "أكثر سطوعًا",
    "باهت",
    "أعلى في السماء",
    "أكثر انخفاضًا في السماء"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "sample_id": "astronomy/test/58",
    "subject": "astronomy",
    "subject_category": "STEM",
    "cultural_sensitivity_label": "CS",
    "language": "Arabic"
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
    --datasets global_mmlu_lite \
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
    datasets=['global_mmlu_lite'],
    dataset_args={
        'global_mmlu_lite': {
            # subset_list: ['ar', 'bn', 'cs']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
