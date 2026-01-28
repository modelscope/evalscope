# BroadTwitterCorpus

## Overview

BroadTwitterCorpus is a dataset of tweets collected over stratified times, places, and social uses. The goal is to represent a broad range of activities, giving a dataset more representative of the language used in this hardest of social media formats to process.

## Task Description

- **Task Type**: Social Media Named Entity Recognition (NER)
- **Input**: Diverse Twitter text (tweets)
- **Output**: Identified entity spans with types
- **Domain**: Social media, diverse contexts

## Key Features

- Stratified sampling across times, places, and uses
- Representative of diverse Twitter language
- Addresses challenges in social media NER
- Three standard NER entity types (PER, ORG, LOC)
- Useful for robust social media NLP evaluation

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `broad_twitter_corpus` |
| **Dataset ID** | [extraordinarylab/broad-twitter-corpus](https://modelscope.cn/datasets/extraordinarylab/broad-twitter-corpus/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,000 |
| Prompt Length (Mean) | 2341.71 chars |
| Prompt Length (Min/Max) | 2246 / 2398 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "62394bfa",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nI hate the words chunder , vomit and puke . BUUH .\n\nOutput:\n<response>I hate the words chunder , vomit and puke . BUUH .</response>\n\nInput:\n♥ . . ) ) ( ♫ . ( ړײ ) ♫ . ♥ . « ▓ » ♥ . ♫ ... [TRUNCATED] ... planations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\n@ colgo hey , congrats to you and the team ! Always worth a read :)\n"
    }
  ],
  "target": "<response><person>@</person> <person>colgo</person> hey , congrats to you and the team ! Always worth a read :)</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "@",
      "colgo",
      "hey",
      ",",
      "congrats",
      "to",
      "you",
      "and",
      "the",
      "team",
      "!",
      "Always",
      "worth",
      "a",
      "read",
      ":)"
    ],
    "ner_tags": [
      "B-PER",
      "B-PER",
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
    --datasets broad_twitter_corpus \
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
    datasets=['broad_twitter_corpus'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


