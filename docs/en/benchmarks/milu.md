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
- Each language is a separate HF dataset config, loaded independently

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split)
- Use `subset_list` to evaluate specific languages (e.g., `['Hindi', 'Tamil']`)
- Requires access to the gated `ai4bharat/MILU` dataset — accept the dataset terms on
  huggingface.co and set `HF_TOKEN` (or run `huggingface-cli login`) before evaluating.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `milu` |
| **Dataset ID** | [ai4bharat/MILU](https://modelscope.cn/datasets/ai4bharat/MILU/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


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
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
