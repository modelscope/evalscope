# MMLU-Indic


## Overview

MMLU-Indic is Sarvam AI's translation of the MMLU (Massive Multitask Language Understanding) benchmark into
10 Indic languages, each in native script and a romanized (Latin transliteration) variant, plus the original
English. It is the fourth of Sarvam's translated academic benchmarks, alongside `arc_indic`, `triviaqa_indic`
and `indic_boolq`.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: Question with answer choices in one of 21 language/script variants
- **Output**: Correct answer letter (A-D)
- **Domain**: The 57 MMLU subjects (STEM, humanities, social sciences and others)

## Key Features

- 21 subsets: English, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu
  in native script, and the 10 Indic languages again as `<lang>_roman`
- Native-script subsets have the full MMLU test set (14,042 questions) and its 285-question dev set as the
  `validation` split; the romanized subsets have 13,096-14,005 test questions and no `validation` split
- The dataset does not keep MMLU's subject labels, so results are reported per language only
- Four English test rows list three choices while their answer index points to a missing fourth choice; they are
  skipped at load time. Six other English rows have three valid choices and are kept as three-option questions

## Evaluation Notes

- Default configuration uses **0-shot** evaluation on the `test` split, with accuracy as the metric
- Few-shot examples come from the `validation` split, available for English and the 10 native-script Indic
  subsets. The `_roman` subsets have no `validation` split and reject few-shot evaluation
- Sarvam reports zero-shot results with an "Indic Average": the mean of the 10 native-script Indic subsets,
  excluding English and the romanized subsets. This is a reporting convention for comparison with Sarvam,
  not an additional aggregate computed by this adapter
- Use `subset_list` to evaluate specific languages or scripts (e.g., `['hi', 'hi_roman']`)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mmlu_indic` |
| **Dataset ID** | [sarvamai/mmlu-indic](https://modelscope.cn/datasets/sarvamai/mmlu-indic/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 293,179 |
| Prompt Length (Mean) | 606.06 chars |
| Prompt Length (Min/Max) | 211 / 8204 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en` | 14,038 | 648.16 | 223 | 5050 |
| `bn` | 14,042 | 582.21 | 223 | 6027 |
| `gu` | 14,042 | 578.53 | 223 | 6000 |
| `hi` | 14,042 | 592.55 | 223 | 6007 |
| `kn` | 14,042 | 615.7 | 223 | 8204 |
| `ml` | 14,042 | 610.76 | 223 | 6054 |
| `mr` | 14,042 | 585.12 | 224 | 7089 |
| `or` | 14,042 | 555.12 | 223 | 6004 |
| `pa` | 14,042 | 593.54 | 223 | 6012 |
| `ta` | 14,042 | 653.79 | 223 | 6394 |
| `te` | 14,042 | 598.43 | 223 | 5989 |
| `bn_roman` | 13,894 | 572.01 | 211 | 3562 |
| `gu_roman` | 14,004 | 592.19 | 211 | 6557 |
| `hi_roman` | 13,913 | 612.57 | 211 | 5139 |
| `kn_roman` | 14,005 | 632.05 | 211 | 4740 |
| `ml_roman` | 13,991 | 643.88 | 216 | 7117 |
| `mr_roman` | 13,904 | 595.36 | 211 | 4486 |
| `or_roman` | 13,979 | 602.7 | 211 | 4536 |
| `pa_roman` | 13,946 | 602.95 | 211 | 4622 |
| `ta_roman` | 13,096 | 639.27 | 216 | 3895 |
| `te_roman` | 13,989 | 622.46 | 211 | 5617 |

## Sample Example

**Subset**: `en`

```json
{
  "input": [
    {
      "id": "36d1bb8c",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nUnlike most other early civilizations, Minoan culture shows little evidence of:\n\nA) trade.\nB) warfare.\nC) the development of a common religion.\nD) conspicuous consumption by elites."
    }
  ],
  "choices": [
    "trade.",
    "warfare.",
    "the development of a common religion.",
    "conspicuous consumption by elites."
  ],
  "target": "D",
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
    --datasets mmlu_indic \
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
    datasets=['mmlu_indic'],
    dataset_args={
        'mmlu_indic': {
            # subset_list: ['en', 'bn', 'gu']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
