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
- Use `subset_list` to evaluate specific languages (e.g., `['hi', 'ta']`)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `triviaqa_indic` |
| **Dataset ID** | [sarvamai/trivia-qa-indic-mcq](https://modelscope.cn/datasets/sarvamai/trivia-qa-indic-mcq/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


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
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
