# TriviaQA-Indic-MCQ


## Overview

TriviaQA-Indic-MCQ reformats TriviaQA trivia questions as 4-way multiple-choice questions, translated
into 10 Indic languages plus English, for evaluating multilingual world-knowledge recall.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Trivia Question Answering
- **Input**: Trivia question with 4 answer choices in one of 11 languages
- **Output**: Correct answer letter
- **Languages**: Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (validation split, the only split available)
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`), or `limit` to cap sample
  count — the full default run is ~18k samples per language across all 11 languages (~198k total)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `triviaqa_indic` |
| **Dataset ID** | [sarvamai/trivia-qa-indic-mcq](https://modelscope.cn/datasets/sarvamai/trivia-qa-indic-mcq/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 197,384 |
| Prompt Length (Mean) | 353.95 chars |
| Prompt Length (Min/Max) | 247 / 1267 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `bn` | 17,944 | 351.81 | 253 | 1131 |
| `en` | 17,944 | 347.69 | 256 | 1158 |
| `gu` | 17,944 | 347.73 | 247 | 1117 |
| `hi` | 17,944 | 349.05 | 258 | 1141 |
| `kn` | 17,944 | 359.89 | 253 | 1164 |
| `ml` | 17,944 | 366.11 | 257 | 1267 |
| `mr` | 17,944 | 350.9 | 253 | 1078 |
| `or` | 17,944 | 349.44 | 255 | 941 |
| `pa` | 17,944 | 345.48 | 253 | 1157 |
| `ta` | 17,944 | 368.55 | 248 | 1198 |
| `te` | 17,944 | 356.75 | 254 | 1259 |

## Sample Example

**Subset**: `bn`

```json
{
  "input": [
    {
      "id": "eff7630c",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nচিপমঙ্কসের পিছনে লোকটি কে ছিল?\n\nA) ডেভিড সেভিল\nB) জাগরেব শহর - ক্রোয়েশিয়া প্রজাতন্ত্র\nC) পবিত্র ক্রুসেড\nD) উপাদান (অ্যালবাম)"
    }
  ],
  "choices": [
    "ডেভিড সেভিল",
    "জাগরেব শহর - ক্রোয়েশিয়া প্রজাতন্ত্র",
    "পবিত্র ক্রুসেড",
    "উপাদান (অ্যালবাম)"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
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
    --datasets triviaqa_indic \
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
    datasets=['triviaqa_indic'],
    dataset_args={
        'triviaqa_indic': {
            # subset_list: ['bn', 'en', 'gu']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
