# DrivelologyNarrativeWriting

## Overview

Drivelology Narrative Writing evaluates models' ability to generate detailed descriptions illustrating the implicit narrative of "drivelology" text - linguistic utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.

## Task Description

- **Task Type**: Narrative Generation and Evaluation
- **Input**: Drivelology text sample
- **Output**: Generated narrative description explaining implicit meaning
- **Domain**: Linguistic analysis, narrative generation

## Key Features

- Tests narrative explanation generation ability
- Requires understanding of layered linguistic meanings
- LLM-as-judge evaluation against reference narratives
- Likert scale scoring (1-5) for match quality
- Tests depth of linguistic and cultural understanding

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM-as-judge for evaluation
- Metrics: Average Likert score (1-5 scale)
- Evaluates relevance, accuracy, depth, and detail of generated narratives

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `drivel_writing` |
| **Dataset ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `Reasoning` |
| **Metrics** | `bert_score`, `gpt_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 600 |
| Prompt Length (Mean) | 313.18 chars |
| Prompt Length (Min/Max) | 256 / 717 chars |

## Sample Example

**Subset**: `narrative-writing-english`

```json
{
  "input": [
    {
      "id": "f47953a9",
      "content": [
        {
          "text": "You need to first read and understand the text given. Generate a detailed description to illustrate the implicit narrative of the text.\n\nPlease provide your response in English, with a clear and comprehensive explanation of the narrative.\n\nText: 後天的努力比什麼都重要，所以今天和明天休息。"
        }
      ]
    }
  ],
  "target": "This creates a paradoxical tone, as it acknowledges the value of diligence but simultaneously advocates for procrastination. The underlying message could reflect a lighthearted take on balancing work and rest or even poking fun at the tendency to delay responsibilities.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "text": "後天的努力比什麼都重要，所以今天和明天休息。",
    "reference_narrative": "This creates a paradoxical tone, as it acknowledges the value of diligence but simultaneously advocates for procrastination. The underlying message could reflect a lighthearted take on balancing work and rest or even poking fun at the tendency to delay responsibilities."
  }
}
```

## Prompt Template

**Prompt Template:**
```text
You need to first read and understand the text given. Generate a detailed description to illustrate the implicit narrative of the text.

Please provide your response in English, with a clear and comprehensive explanation of the narrative.

Text: {text}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drivel_writing \
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
    datasets=['drivel_writing'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


