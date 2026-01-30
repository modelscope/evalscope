# CoNLL2003


## Overview

CoNLL-2003 is a classic Named Entity Recognition (NER) benchmark introduced at the Conference on Computational Natural Language Learning 2003. It contains news articles annotated with four entity types.

## Task Description

- **Task Type**: Named Entity Recognition (NER)
- **Input**: Text with entities to identify
- **Output**: Entity spans with type labels
- **Entity Types**: Person (PER), Organization (ORG), Location (LOC), Miscellaneous (MISC)

## Key Features

- Standard NER benchmark with well-defined entity types
- News domain text with high annotation quality
- Four entity categories with clear definitions
- Supports few-shot evaluation
- Comprehensive metrics (precision, recall, F1, accuracy)

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: **Precision**, **Recall**, **F1 Score**, **Accuracy**
- Train split: **train**, Eval split: **test**
- Entity types mapped to human-readable names


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `conll2003` |
| **Dataset ID** | [extraordinarylab/conll2003](https://modelscope.cn/datasets/extraordinarylab/conll2003/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,453 |
| Prompt Length (Mean) | 2733.98 chars |
| Prompt Length (Min/Max) | 2663 / 3275 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "c8d2b130",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nEU rejects German call to boycott British lamb .\n\nOutput:\n<response><organization>EU</organization> rejects <miscellaneous>German</miscellaneous> call to boycott <miscellaneous>Briti ... [TRUNCATED] ... include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nSOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .\n"
    }
  ],
  "target": "<response>SOCCER - <location>JAPAN</location> GET LUCKY WIN , <person>CHINA</person> IN SURPRISE DEFEAT .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "SOCCER",
      "-",
      "JAPAN",
      "GET",
      "LUCKY",
      "WIN",
      ",",
      "CHINA",
      "IN",
      "SURPRISE",
      "DEFEAT",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "B-LOC",
      "O",
      "O",
      "O",
      "O",
      "B-PER",
      "O",
      "O",
      "O",
      "O"
    ]
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

<details>
<summary>Few-shot Template</summary>

```text
Here are some examples of named entity recognition:

{fewshot}

You are a named entity recognition system that identifies the following entity types:
{entities}

Process the provided text and mark all named entities with XML-style tags.

For example:
<person>John Smith</person> works at <organization>Google</organization> in <location>Mountain View</location>.

Available entity tags: {entity_list}

INSTRUCTIONS:
1. Wrap your entire response in <response>...</response> tags.
2. Inside these tags, include the original text with entity tags inserted.
3. Do not change the original text in any way (preserve spacing, punctuation, case, etc.).
4. Tag ALL entities you can identify using the exact tag names provided.
5. Do not include explanations, just the tagged text.
6. If entity spans overlap, choose the most specific entity type.
7. Ensure every opening tag has a matching closing tag.

Text to process:
{text}

```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets conll2003 \
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
    datasets=['conll2003'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


