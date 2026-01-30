# HarveyNER

## Overview

HarveyNER is a dataset with fine-grained locations annotated in tweets, collected during Hurricane Harvey. It presents unique challenges with complex and long location mentions in informal crisis-related descriptions.

## Task Description

- **Task Type**: Crisis-Domain Location Named Entity Recognition (NER)
- **Input**: Hurricane Harvey-related tweets
- **Output**: Fine-grained location entity spans
- **Domain**: Crisis communication, disaster response

## Key Features

- Fine-grained location annotations in tweets
- Complex and long location mentions
- Informal crisis-related text
- Four location entity types (AREA, POINT, RIVER, ROAD)
- Useful for disaster response NLP applications

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: AREA, POINT, RIVER, ROAD

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `harvey_ner` |
| **Dataset ID** | [extraordinarylab/harvey-ner](https://modelscope.cn/datasets/extraordinarylab/harvey-ner/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,303 |
| Prompt Length (Mean) | 3018.97 chars |
| Prompt Length (Min/Max) | 2909 / 3199 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "629de70e",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nJust received word that UHVictoria and Bayou Oaks residents are in need of blankets , pillows , and clothes ( men & amp ; women ) . ( PT1 )\n\nOutput:\n<response>Just received word that ... [TRUNCATED] ... lap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nBREAKING : One firefighter injured after a fire / apparent explosion at the Lone Star Legal Aid Services in Downtown Houston\n"
    }
  ],
  "target": "<response>BREAKING : One firefighter injured after a fire / apparent explosion at the <point>Lone Star Legal Aid Services in Downtown Houston</point></response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "BREAKING",
      ":",
      "One",
      "firefighter",
      "injured",
      "after",
      "a",
      "fire",
      "/",
      "apparent",
      "explosion",
      "at",
      "the",
      "Lone",
      "Star",
      "Legal",
      "Aid",
      "Services",
      "in",
      "Downtown",
      "Houston"
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT",
      "I-POINT"
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
    --datasets harvey_ner \
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
    datasets=['harvey_ner'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


