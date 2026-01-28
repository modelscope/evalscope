# MIT-Restaurant

## Overview

The MIT-Restaurant dataset is a collection of restaurant review text specifically curated for training and testing NLP models for Named Entity Recognition. It contains sentences from real reviews with annotations in BIO format.

## Task Description

- **Task Type**: Restaurant Domain Named Entity Recognition (NER)
- **Input**: Restaurant review text and queries
- **Output**: Identified restaurant-related entity spans
- **Domain**: Food service, restaurant reviews, dialogue systems

## Key Features

- Real restaurant review sentences
- BIO format annotations
- Eight restaurant-specific entity types
- Useful for food service domain NLP
- Adapted for conversational AI applications

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: AMENITY, CUISINE, DISH, HOURS, LOCATION, PRICE, RATING, RESTAURANT_NAME

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mit_restaurant` |
| **Dataset ID** | [extraordinarylab/mit-restaurant](https://modelscope.cn/datasets/extraordinarylab/mit-restaurant/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,521 |
| Prompt Length (Mean) | 2383.97 chars |
| Prompt Length (Min/Max) | 2338 / 2474 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "9d5a77f1",
      "content": "Here are some examples of named entity recognition:\n\nInput:\ncan you find me the cheapest mexican restaurant nearby\n\nOutput:\n<response>can you find me the <price>cheapest</price> <cuisine>mexican</cuisine> restaurant <location>nearby</location ... [TRUNCATED] ... mes provided.\n5. Do not include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\na four star restaurant with a bar\n"
    }
  ],
  "target": "<response>a <rating>four star</rating> restaurant <location>with a</location> <amenity>bar</amenity></response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "a",
      "four",
      "star",
      "restaurant",
      "with",
      "a",
      "bar"
    ],
    "ner_tags": [
      "O",
      "B-RATING",
      "I-RATING",
      "O",
      "B-LOCATION",
      "I-LOCATION",
      "B-AMENITY"
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
    --datasets mit_restaurant \
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
    datasets=['mit_restaurant'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


