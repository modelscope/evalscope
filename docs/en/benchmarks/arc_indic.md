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
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `validation` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

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
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
