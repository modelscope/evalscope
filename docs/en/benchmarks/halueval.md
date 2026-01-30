# HaluEval

## Overview

HaluEval is a large collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognizing hallucination. It provides a comprehensive benchmark for assessing model reliability and factual accuracy.

## Task Description

- **Task Type**: Hallucination Detection
- **Input**: Context/knowledge + response to judge
- **Output**: YES (hallucination) or NO (factual)
- **Domains**: Dialogue, QA, Summarization

## Key Features

- Three evaluation categories:
  - `dialogue_samples`: Hallucination in conversational responses
  - `qa_samples`: Hallucination in question answering
  - `summarization_samples`: Hallucination in document summaries
- Both generated and human-annotated samples
- Tests model's ability to detect factual inconsistencies
- Requires reasoning about knowledge-response alignment

## Evaluation Notes

- Default evaluation uses **zero-shot** (no few-shot examples)
- Multiple metrics computed:
  - **Accuracy**: Overall correct judgments
  - **Precision**: True positives among positive predictions
  - **Recall**: True positives among actual positives
  - **F1 Score**: Harmonic mean of precision and recall
  - **Yes Ratio**: Proportion of YES predictions
- Binary YES/NO judgment format

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `halueval` |
| **Dataset ID** | [evalscope/HaluEval](https://modelscope.cn/datasets/evalscope/HaluEval/summary) |
| **Paper** | N/A |
| **Tags** | `Hallucination`, `Knowledge`, `Yes/No` |
| **Metrics** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `data` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 30,000 |
| Prompt Length (Mean) | 4832.18 chars |
| Prompt Length (Min/Max) | 2463 / 16078 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `dialogue_samples` | 10,000 | 3563.69 | 3169 | 4200 |
| `qa_samples` | 10,000 | 2811.83 | 2463 | 4004 |
| `summarization_samples` | 10,000 | 8121.02 | 4932 | 16078 |

## Sample Example

**Subset**: `dialogue_samples`

```json
{
  "input": [
    {
      "id": "a99406f3",
      "content": [
        {
          "text": "I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallu ... [TRUNCATED] ...  do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film. \n#Response#: I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.\n#Your Judgement#:"
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "yes"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets halueval \
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
    datasets=['halueval'],
    dataset_args={
        'halueval': {
            # subset_list: ['dialogue_samples', 'qa_samples', 'summarization_samples']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


