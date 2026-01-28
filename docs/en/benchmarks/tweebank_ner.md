# TweeBankNER


## Overview

Tweebank-NER is an English Twitter corpus created by annotating the syntactically-parsed Tweebank V2 with four types of named entities: Person, Organization, Location, and Miscellaneous. It addresses NER challenges in informal social media text.

## Task Description

- **Task Type**: Social Media Named Entity Recognition (NER)
- **Input**: Twitter text (tweets)
- **Output**: Identified entity spans with types
- **Domain**: Social media, informal text

## Key Features

- Based on Tweebank V2 syntactic annotations
- Four standard NER entity types
- Addresses informal language challenges
- Handles Twitter-specific text features (hashtags, mentions)
- Useful for social media NLP applications

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC, MISC


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tweebank_ner` |
| **Dataset ID** | [extraordinarylab/tweebank-ner](https://modelscope.cn/datasets/extraordinarylab/tweebank-ner/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,201 |
| Prompt Length (Mean) | 2322.67 chars |
| Prompt Length (Min/Max) | 2250 / 2398 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "46ad4b5f",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nRT @USER2362 : Farmall Heart Of The Holidays Tabletop Christmas Tree With Lights And Motion URL1087 #Holiday #Gifts\n\nOutput:\n<response>RT @USER2362 : <organization>Farmall</organizat ... [TRUNCATED] ... include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\n@USER1812 No , I 'm not . It 's definitely not a rapper .\n"
    }
  ],
  "target": "<response>@USER1812 No , I 'm not . It 's definitely not a rapper .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "@USER1812",
      "No",
      ",",
      "I",
      "'m",
      "not",
      ".",
      "It",
      "'s",
      "definitely",
      "not",
      "a",
      "rapper",
      "."
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
    --datasets tweebank_ner \
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
    datasets=['tweebank_ner'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


