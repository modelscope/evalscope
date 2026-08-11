# KINA


## Overview

KINA (Knowledge Index of Noah's Ark) is a high-density multidisciplinary knowledge benchmark for evaluating whether large language models can solve expert-level questions across 261 fine-grained disciplines. It is the first benchmark to incorporate disciplinary representativeness as a core design principle.

## Task Description

- **Task Type**: Multiple-Choice Question Answering (MCQ)
- **Input**: A discipline-specific question with up to 10 lettered options (A–J)
- **Output**: A single correct answer letter (A–J)
- **Domains**: 261 disciplines spanning Agronomy, Medicine, Engineering, Humanities, Natural Sciences, and more

## Key Features

- 899 test questions covering 261 fine-grained disciplines
- Each question has a unique correct answer among up to 10 options (A–J)
- Includes per-option explanations for training / analysis (not shown to the model)
- Designed to test deep domain knowledge, not retrieval or commonsense reasoning
- Introduced at 2077AI with a focus on disciplinary representativeness

## Evaluation Notes

- Default evaluation uses the **test** split (899 samples)
- Primary metric: **Accuracy** (`accuracy`) — Pass@1 for single-inference mode
- 0-shot Chain-of-Thought (CoT) evaluation, answer extracted from ``ANSWER: [LETTER]`` marker
- Discipline metadata is stored per-sample and available in review output; no per-discipline subset grouping
- [GitHub](https://github.com/weihao1115/KINA-Benchmark)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `kina` |
| **Dataset ID** | [evalscope/KINA](https://modelscope.cn/datasets/evalscope/KINA/summary) |
| **Paper** | [Paper](https://www.2077ai.com/kina) |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets kina \
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
    datasets=['kina'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
