# MILU


## Overview

MILU (Multi-task Indic Language Understanding Benchmark) is a comprehensive evaluation dataset for
assessing LLM performance across 11 Indic languages. It spans 8 domains and 41 subjects, combining
translated general-knowledge questions with culturally specific Indian content.

## Task Description

- **Task Type**: Multilingual Multiple-Choice Question Answering
- **Input**: Question with four answer choices in one of 11 languages
- **Output**: Single correct answer letter
- **Languages**: English, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Key Features

- 8 domains / 41 subjects, including India-specific culture, history, and current affairs
- Native-language questions rather than machine-translated MMLU
- Each language is a separate dataset config, loaded independently

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split)
- Use `subset_list` to evaluate specific languages (e.g., `['Hindi', 'Tamil']`), or `limit` to cap
  sample count — evaluating all 11 languages' full test splits is a large run
- Set `few_shot_num` > 0 to enable few-shot prompting; examples are drawn from the `validation` split
- Loads from ModelScope by default (evalscope's default `dataset_hub`), where this dataset is public
  and needs no token. If you explicitly set `dataset_hub` to `huggingface`, note that
  `ai4bharat/MILU` is gated there — accept the dataset terms on huggingface.co and set `HF_TOKEN`
  (or run `huggingface-cli login`) first


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `milu` |
| **Dataset ID** | [ai4bharat/MILU](https://modelscope.cn/datasets/ai4bharat/MILU/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 79,608 |
| Prompt Length (Mean) | 377.16 chars |
| Prompt Length (Min/Max) | 223 / 2110 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `English` | 13,535 | 397.01 | 227 | 1930 |
| `Bengali` | 6,637 | 359.93 | 232 | 1828 |
| `Gujarati` | 4,826 | 359.36 | 230 | 1785 |
| `Hindi` | 14,831 | 367.43 | 229 | 1907 |
| `Kannada` | 6,234 | 364.45 | 229 | 1753 |
| `Malayalam` | 4,321 | 388.2 | 239 | 2110 |
| `Marathi` | 6,924 | 394.85 | 223 | 1888 |
| `Odia` | 4,525 | 366.63 | 238 | 1825 |
| `Punjabi` | 4,099 | 364.93 | 234 | 1874 |
| `Tamil` | 6,372 | 382.22 | 230 | 1934 |
| `Telugu` | 7,304 | 384.05 | 233 | 1806 |

## Sample Example

**Subset**: `English`

```json
{
  "input": [
    {
      "id": "84726982",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nBakelite is what type of polymer?\n\nA) Thermosetting polymer\nB) Thermoplastic polymer\nC) Fibre\nD) Elastomer"
    }
  ],
  "choices": [
    "Thermosetting polymer",
    "Thermoplastic polymer",
    "Fibre",
    "Elastomer"
  ],
  "target": "A",
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
    --datasets milu \
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
    datasets=['milu'],
    dataset_args={
        'milu': {
            # subset_list: ['English', 'Bengali', 'Gujarati']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
