# MIT-Movie-Trivia

## Overview

The MIT-Movie-Trivia dataset, originally created for slot filling in movie domain dialogues, has been modified for NER by merging and filtering slot types. It tests recognition of movie-related entities in conversational queries.

## Task Description

- **Task Type**: Movie Domain Named Entity Recognition (NER)
- **Input**: Movie-related conversational queries
- **Output**: Identified movie entity spans
- **Domain**: Entertainment, movie trivia, dialogue systems

## Key Features

- Movie domain conversational queries
- Rich entity type coverage for movies
- Adapted from slot filling task
- Twelve entity types covering movie attributes
- Useful for entertainment domain NLP

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: ACTOR, AWARD, CHARACTER_NAME, DIRECTOR, GENRE, OPINION, ORIGIN, PLOT, QUOTE, RELATIONSHIP, SOUNDTRACK, YEAR

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mit_movie_trivia` |
| **Dataset ID** | [extraordinarylab/mit-movie-trivia](https://modelscope.cn/datasets/extraordinarylab/mit-movie-trivia/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,953 |
| Prompt Length (Mean) | 3309.63 chars |
| Prompt Length (Min/Max) | 3216 / 3565 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "c33a15eb",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nwhat 1995 romantic comedy film starred michael douglas as a u s head of state looking for love\n\nOutput:\n<response>what <year>1995</year> <genre>romantic comedy</genre> film starred < ... [TRUNCATED] ... If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\ni need that movie which involves aliens invading earth in a particular united states place in california\n"
    }
  ],
  "target": "<response>i need that movie which involves <plot>aliens invading earth in a particular united states place in california</plot></response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "i",
      "need",
      "that",
      "movie",
      "which",
      "involves",
      "aliens",
      "invading",
      "earth",
      "in",
      "a",
      "particular",
      "united",
      "states",
      "place",
      "in",
      "california"
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT",
      "I-PLOT"
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
    --datasets mit_movie_trivia \
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
    datasets=['mit_movie_trivia'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


