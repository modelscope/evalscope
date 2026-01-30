# Med-MCQA

## Overview

MedMCQA is a large-scale multiple-choice question answering dataset designed to address real-world medical entrance exam questions. It contains over 194K questions covering diverse medical topics from Indian medical entrance examinations (AIIMS, NEET-PG).

## Task Description

- **Task Type**: Medical Knowledge Multiple-Choice QA
- **Input**: Medical question with 4 answer choices
- **Output**: Correct answer letter
- **Domain**: Clinical medicine, basic sciences, healthcare

## Key Features

- Over 194,000 medical exam questions
- Real questions from AIIMS and NEET-PG exams
- 21 medical subjects covered (anatomy, pharmacology, pathology, etc.)
- Expert-verified correct answers with explanations
- Tests medical knowledge comprehension and clinical reasoning

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on validation split
- Simple accuracy metric
- Train split available for few-shot learning

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `med_mcqa` |
| **Dataset ID** | [extraordinarylab/medmcqa](https://modelscope.cn/datasets/extraordinarylab/medmcqa/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,183 |
| Prompt Length (Mean) | 374.31 chars |
| Prompt Length (Min/Max) | 232 / 1004 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "9bf68985",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich of the following is not true for myelinated ner ... [TRUNCATED] ... ugh myelinated fibers is slower than non-myelinated fibers\nB) Membrane currents are generated at nodes of Ranvier\nC) Saltatory conduction of impulses is seen\nD) Local anesthesia is effective only when the nerve is not covered by myelin sheath"
    }
  ],
  "choices": [
    "Impulse through myelinated fibers is slower than non-myelinated fibers",
    "Membrane currents are generated at nodes of Ranvier",
    "Saltatory conduction of impulses is seen",
    "Local anesthesia is effective only when the nerve is not covered by myelin sheath"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {}
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
    --datasets med_mcqa \
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
    datasets=['med_mcqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


