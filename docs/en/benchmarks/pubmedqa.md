# PubMedQA

## Overview

PubMedQA is a biomedical question answering dataset designed to evaluate models' ability to reason over biomedical research texts. It contains questions derived from PubMed abstracts with yes/no/maybe answers.

## Task Description

- **Task Type**: Biomedical Question Answering (Yes/No/Maybe)
- **Input**: PubMed abstract with a research question
- **Output**: Answer (YES, NO, or MAYBE)
- **Domain**: Biomedical research, scientific literature

## Key Features

- Questions from real PubMed research abstracts
- Three-way classification (YES/NO/MAYBE)
- Tests scientific reasoning and comprehension
- Expert-annotated with reasoning explanations
- Useful for biomedical AI evaluation

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, Yes Ratio, Maybe Ratio
- Aggregation method: F1-score (macro average)
- Answers should be YES, NO, or MAYBE without explanation

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `pubmedqa` |
| **Dataset ID** | [extraordinarylab/pubmed-qa](https://modelscope.cn/datasets/extraordinarylab/pubmed-qa/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `Yes/No` |
| **Metrics** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio`, `maybe_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `f1` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Prompt Length (Mean) | 1514.46 chars |
| Prompt Length (Min/Max) | 451 / 2883 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "f17dedfd",
      "content": [
        {
          "text": "Abstract: Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longit ... [TRUNCATED] ... ntrols, and that displayed mitochondrial dynamics similar to that of non-PCD cells.\n\nQuestion: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\nPlease answer YES or NO or MAYBE without an explanation."
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "yes",
    "reasoning": "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. To the best of our knowledge, this is the first report of ... [TRUNCATED] ... PCD. Also, for the first time, we have shown the feasibility for the use of CsA in a whole plant system. Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant."
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
Please answer YES or NO or MAYBE without an explanation.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets pubmedqa \
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
    datasets=['pubmedqa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


