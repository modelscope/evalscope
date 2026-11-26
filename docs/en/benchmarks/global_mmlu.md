# Global-MMLU


## Overview

Global-MMLU (Cohere Labs, 2024) is MMLU in 42 languages. It combines machine translations with professional
translations and community post-edits, and labels part of the questions as culturally sensitive (CS) or culturally
agnostic (CA), so a model's multilingual knowledge can be told apart from its Western-centric knowledge.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: An MMLU question with four options in one of 42 languages
- **Output**: The letter of the correct option (A-D)
- **Domain**: The 57 MMLU subjects

## Key Features

- 42 subsets, one per language, each with all 14,042 MMLU test questions (589,764 in total) and the 285 `dev`
  questions (five per subject)
- 2,850 questions per language carry a cultural sensitivity label: 792 CS and 2,058 CA; the rest are unannotated
- 39 test questions (0.007%, mostly Amharic, Hindi and Vietnamese) lost an option in translation; they are kept
  with that option empty, as in lm-evaluation-harness, so every language keeps all 14,042 questions. In 12 of
  them the lost option is the correct one
- For a smaller, fully human-translated set in 23 languages, use `global_mmlu_lite`

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
| Prompt Length (Mean) | 3049.29 chars |
| Prompt Length (Min/Max) | 558 / 16835 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `am` | 14,042 | 2144.97 | 749 | 8937 |
| `ar` | 14,042 | 2718.9 | 830 | 11584 |
| `bn` | 14,042 | 2986.99 | 850 | 13433 |
| `cs` | 14,042 | 2956.69 | 863 | 13127 |
| `de` | 14,042 | 3570.78 | 1007 | 16685 |
| `el` | 14,042 | 3447.4 | 1003 | 15859 |
| `en` | 14,042 | 3179.89 | 949 | 14594 |
| `es` | 14,042 | 3469.11 | 977 | 15556 |
| `fa` | 14,042 | 2773.82 | 821 | 12018 |
| `fil` | 14,042 | 3642.27 | 997 | 16835 |
| `fr` | 14,042 | 3554.32 | 1015 | 15947 |
| `ha` | 14,042 | 3184.16 | 890 | 14093 |
| `he` | 14,042 | 2412.8 | 813 | 10445 |
| `hi` | 14,042 | 3087.61 | 900 | 13884 |
| `id` | 14,042 | 3447.94 | 938 | 15651 |
| `ig` | 14,042 | 2999.44 | 896 | 13562 |
| `it` | 14,042 | 3482.26 | 984 | 15316 |
| `ja` | 14,042 | 1571.26 | 627 | 6229 |
| `ko` | 14,042 | 1733.25 | 661 | 7097 |
| `ky` | 14,042 | 3122.1 | 936 | 14341 |
| `lt` | 14,042 | 3054.74 | 921 | 13664 |
| `mg` | 14,042 | 3568.98 | 1042 | 16250 |
| `ms` | 14,042 | 3346.5 | 947 | 15393 |
| `ne` | 14,042 | 2894.82 | 861 | 13033 |
| `nl` | 14,042 | 3488.92 | 993 | 16074 |
| `ny` | 14,042 | 3308.46 | 1012 | 15235 |
| `pl` | 14,042 | 3276.01 | 909 | 14503 |
| `pt` | 14,042 | 3295.69 | 970 | 14620 |
| `ro` | 14,042 | 3351.94 | 932 | 14854 |
| `ru` | 14,042 | 3319.02 | 920 | 14635 |
| `si` | 14,042 | 2959.09 | 881 | 13125 |
| `sn` | 14,042 | 3218.24 | 962 | 14630 |
| `so` | 14,042 | 3445.62 | 944 | 15603 |
| `sr` | 14,042 | 2978.02 | 901 | 13118 |
| `sv` | 14,042 | 3117.7 | 924 | 14506 |
| `sw` | 14,042 | 3215.31 | 941 | 14343 |
| `te` | 14,042 | 3045.32 | 900 | 13672 |
| `tr` | 14,042 | 3094.77 | 904 | 14003 |
| `uk` | 14,042 | 3113.36 | 908 | 13695 |
| `vi` | 14,042 | 3253.18 | 977 | 15261 |
| `yo` | 14,042 | 3004.3 | 938 | 13722 |
| `zh` | 14,042 | 1234.4 | 558 | 4397 |

## Sample Example

**Subset**: `am`

```json
{
  "input": [
    {
      "id": "c45ee91e",
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
