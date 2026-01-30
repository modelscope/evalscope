# TweetNER7


## Overview

TweetNER7 is a large-scale NER dataset featuring over 11,000 tweets from 2019-2021, annotated with seven entity types to facilitate the study of short-term temporal shifts in social media language.

## Task Description

- **Task Type**: Social Media Named Entity Recognition (NER)
- **Input**: Twitter text from 2019-2021
- **Output**: Identified entity spans with seven types
- **Domain**: Social media, temporal language variation

## Key Features

- Over 11,000 annotated tweets
- Seven diverse entity types
- Spans 2019-2021 for temporal analysis
- Studies language shift in social media
- Rich entity type coverage for social content

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: CORPORATION, CREATIVE_WORK, EVENT, GROUP, LOCATION, PERSON, PRODUCT


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tweet_ner_7` |
| **Dataset ID** | [extraordinarylab/tweet-ner-7](https://modelscope.cn/datasets/extraordinarylab/tweet-ner-7/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,383 |
| Prompt Length (Mean) | 3364.87 chars |
| Prompt Length (Min/Max) | 3226 / 3630 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "cde2f5cb",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nMorning 5km run with {{USERNAME}} for breast cancer awareness # pinkoctober # breastcancerawareness # zalorafit # zalorafitxbnwrc @ The Central Park , Desa Parkcity {{URL}}\n\nOutput:\n ... [TRUNCATED] ...  it 's been an amazing experience . Afro paper : the rolling stone project and I do n't think you know Popin boyz are projects that 's gonna shake the industry . Do n't sleep on {{USERNAME}} {{USERNAME}} {{USERNAME}} Watch out for these guys\n"
    }
  ],
  "target": "<response>Hanging out with the # <group>Popinboyz</group> and it 's been an amazing experience . <creative_work>Afro paper</creative_work> : <creative_work>the rolling stone project</creative_work> and I do n't think you know <group>Popin boyz</group> are projects that 's gonna shake the industry . Do n't sleep on {{USERNAME}} {{USERNAME}} {{USERNAME}} Watch out for these guys</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Hanging",
      "out",
      "with",
      "the",
      "#",
      "Popinboyz",
      "and",
      "it",
      "'s",
      "been",
      "an",
      "amazing",
      "experience",
      ".",
      "Afro",
      "paper",
      ":",
      "the",
      "rolling",
      "stone",
      "project",
      "and",
      "I",
      "do",
      "n't",
      "think",
      "you",
      "know",
      "Popin",
      "boyz",
      "are",
      "projects",
      "that",
      "'s",
      "gonna",
      "shake",
      "the",
      "industry",
      ".",
      "Do",
      "n't",
      "sleep",
      "on",
      "{{USERNAME}}",
      "{{USERNAME}}",
      "{{USERNAME}}",
      "Watch",
      "out",
      "for",
      "these",
      "guys"
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-GROUP",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "O",
      "B-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "I-CREATIVE_WORK",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-GROUP",
      "I-GROUP",
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
    --datasets tweet_ner_7 \
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
    datasets=['tweet_ner_7'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


