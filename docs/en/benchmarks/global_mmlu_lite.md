# Global-MMLU-Lite


## Overview

Global-MMLU-Lite (Cohere Labs) is the balanced, fully human-translated or post-edited subset of Global-MMLU: 200
culturally sensitive (CS) and 200 culturally agnostic (CA) MMLU questions in each of 23 languages.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 23 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: 43 of the 57 MMLU subjects

## Key Features

- 23 subsets, one per language, each with 400 test questions (9,199 evaluated in total, see below) and 215 `dev`
  questions (285 for Czech), five per subject
- The language set differs from Global-MMLU: Welsh, Hungarian, Burmese, Oriya, Slovak, Albanian and Tajik are only in
  the Lite set. The paper's Lite results cover the 15 languages of the first release
- Every question is labeled CS or CA
- Burmese (`my`) test row 202 was released without its question and is skipped, so Burmese has 399 questions
- The Hungarian rows name their subject in Hungarian; few-shot examples are matched on the English subject in
  `sample_id` instead, which every row carries
- Welsh (`cy`) is released without a `dev` split, so it is always evaluated zero-shot

## Evaluation Notes

- Default configuration uses **5-shot** evaluation, as in the paper. The examples for a question are the five `dev`
  questions of its own subject (each subject has exactly five), in the same language
- Every language reports its accuracy on all questions, and separately on the culturally sensitive (`cs`) and
  culturally agnostic (`ca`) questions, as the `cultural_sensitivity` dimension of the accuracy metric
- Languages are grouped by the resource level the dataset card gives them (`high-resource`, `mid-resource`,
  `low-resource`)
- License: [Apache 2.0](https://opensource.org/license/apache-2-0)


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
| Prompt Length (Mean) | 2616.93 chars |
| Prompt Length (Min/Max) | 252 / 16179 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `ar` | 400 | 2469.9 | 995 | 11150 |
| `bn` | 400 | 2687.86 | 987 | 12826 |
| `cs` | 400 | 2680.84 | 958 | 12785 |
| `cy` | 400 | 615.88 | 252 | 2783 |
| `de` | 400 | 3215.65 | 1037 | 16179 |
| `en` | 400 | 2873.45 | 984 | 14330 |
| `es` | 400 | 3127.28 | 1019 | 15071 |
| `fr` | 400 | 3207.98 | 1037 | 15429 |
| `hi` | 400 | 2789.78 | 966 | 13386 |
| `hu` | 400 | 2978.34 | 996 | 14592 |
| `id` | 400 | 3073.09 | 1030 | 15137 |
| `it` | 400 | 3115.3 | 1024 | 15079 |
| `ja` | 400 | 1444.82 | 718 | 6013 |
| `ko` | 400 | 1571.25 | 709 | 6645 |
| `my` | 399 | 3147.42 | 1019 | 15746 |
| `or` | 400 | 2726.38 | 995 | 13442 |
| `pt` | 400 | 2974.54 | 994 | 14253 |
| `sk` | 400 | 2772.8 | 977 | 13133 |
| `sq` | 400 | 3030.3 | 1033 | 14570 |
| `sw` | 400 | 2882.9 | 1046 | 14042 |
| `tg` | 400 | 2955.62 | 981 | 13788 |
| `yo` | 400 | 2697.83 | 984 | 13569 |
| `zh` | 400 | 1151.58 | 624 | 4255 |

## Sample Example

**Subset**: `ar`

```json
{
  "input": [
    {
      "id": "e4c86fc6",
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
