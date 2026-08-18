# ARC-Challenge-Indic


## Overview

ARC-Challenge-Indic is a translation of the AI2 Reasoning Challenge (ARC-Challenge) science
question-answering benchmark into 10 Indic languages, plus the original English set, for evaluating
multilingual scientific reasoning.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Science Question Answering
- **Input**: Science question with answer choices in one of 11 languages
- **Output**: Correct answer letter
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split)
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`)
- Same underlying science-exam questions as `arc` (Challenge split), machine/human translated per language


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `arc_indic` |
| **Dataset ID** | [sarvamai/arc-challenge-indic](https://modelscope.cn/datasets/sarvamai/arc-challenge-indic/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `MultiLingual`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 12,647 |
| Prompt Length (Mean) | 448.01 chars |
| Prompt Length (Min/Max) | 236 / 2053 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `bn` | 1,150 | 432.51 | 242 | 1137 |
| `en` | 1,147 | 454.88 | 253 | 1111 |
| `gu` | 1,150 | 426.57 | 243 | 1098 |
| `hi` | 1,150 | 443.47 | 236 | 1162 |
| `kn` | 1,150 | 456.08 | 245 | 1199 |
| `ml` | 1,150 | 473.31 | 239 | 2053 |
| `mr` | 1,150 | 434.22 | 242 | 1133 |
| `or` | 1,150 | 440.04 | 243 | 1374 |
| `pa` | 1,150 | 443.35 | 236 | 1132 |
| `ta` | 1,150 | 479.12 | 243 | 1295 |
| `te` | 1,150 | 444.53 | 244 | 1172 |

## Sample Example

**Subset**: `bn`

```json
{
  "input": [
    {
      "id": "f750462b",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nএকজন খগোলবিদ পর্যবেক্ষণ করেন যে একটি উল্কা পতনের পরে একটি গ্রহের ঘূর্ণন গতি বেড়ে যায়। ঘূর্ণন বৃদ্ধির ফলে কোন প্রভাবটি সবচেয়ে বেশি সম্ভাব্য?\n\nA) গ্রহের ঘনত্ব কমে যাবে।\nB) গ্রহীয় বছরগুলি আরও দীর্ঘ হবে।\nC) গ্রহের দিনগুলি ছোট হয়ে যাবে।\nD) গ্রহের মাধ্যাকর্ষণ শক্তি আরও বৃদ্ধি পাবে।"
    }
  ],
  "choices": [
    "গ্রহের ঘনত্ব কমে যাবে।",
    "গ্রহীয় বছরগুলি আরও দীর্ঘ হবে।",
    "গ্রহের দিনগুলি ছোট হয়ে যাবে।",
    "গ্রহের মাধ্যাকর্ষণ শক্তি আরও বৃদ্ধি পাবে।"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "Mercury_7175875",
    "language": "Bengali"
  }
}
```

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
    --datasets arc_indic \
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
    datasets=['arc_indic'],
    dataset_args={
        'arc_indic': {
            # subset_list: ['bn', 'en', 'gu']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
