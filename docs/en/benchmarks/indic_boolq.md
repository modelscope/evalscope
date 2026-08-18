# BoolQ-Indic


## Overview

BoolQ-Indic is a translation of the BoolQ yes/no reading-comprehension benchmark into 10 Indic
languages plus English, for evaluating multilingual passage understanding.

## Task Description

- **Task Type**: Multilingual Yes/No Reading Comprehension
- **Input**: Passage + yes/no question in one of 11 languages
- **Output**: `Yes` or `No`
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (validation split)
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`), or `limit` to cap sample
  count — the full default run is 35,970 samples across all 11 languages
- Set `few_shot_num` > 0 to enable few-shot prompting; examples are drawn from the `train` split
- All languages ship in a single dataset config; this adapter reformats by the `language` field


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `indic_boolq` |
| **Dataset ID** | [sarvamai/boolq-indic](https://modelscope.cn/datasets/sarvamai/boolq-indic/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ`, `MultiLingual`, `ReadingComprehension` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 35,970 |
| Prompt Length (Mean) | 822.66 chars |
| Prompt Length (Min/Max) | 275 / 5035 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `bn` | 3,270 | 801.95 | 294 | 2308 |
| `en` | 3,270 | 814.26 | 292 | 5035 |
| `gu` | 3,270 | 793.37 | 283 | 2105 |
| `hi` | 3,270 | 818.47 | 297 | 3078 |
| `kn` | 3,270 | 833.99 | 275 | 2920 |
| `ml` | 3,270 | 869.99 | 294 | 3558 |
| `mr` | 3,270 | 806.68 | 289 | 2593 |
| `or` | 3,270 | 787.68 | 306 | 1482 |
| `pa` | 3,270 | 804.93 | 295 | 1975 |
| `ta` | 3,270 | 904.01 | 297 | 3570 |
| `te` | 3,270 | 813.95 | 284 | 3312 |

## Sample Example

**Subset**: `bn`

```json
{
  "input": [
    {
      "id": "0f04f0f7",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nসকল জৈববস্তুই কমপক্ষে এই ধাপগুলোর মধ্য দিয়ে যায়: এগুলো চা ... [TRUNCATED 1099 chars] ...  বার্কলেতে  ছয়টি পৃথক গবেষণা বিশ্লেষণ করার পর, একটি গবেষণায় উপসংহারে আসা গেছে যে, ভুট্টা থেকে ইথানল উৎপাদনে পেট্রোলিয়ামের ব্যবহার গ্যাসোলিন উৎপাদনের তুলনায় অনেক কম।\n\nQuestion: ইথানল উৎপাদনের চেয়ে  তৈরিতে কি বেশি শক্তি লাগে??\n\nA) Yes\nB) No"
    }
  ],
  "choices": [
    "Yes",
    "No"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "bn",
  "metadata": {
    "language": "Bengali"
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
    --datasets indic_boolq \
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
    datasets=['indic_boolq'],
    dataset_args={
        'indic_boolq': {
            # subset_list: ['bn', 'en', 'gu']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
