# MultiNERD


## Overview

MultiNERD is a large-scale, multilingual, and multi-genre dataset for fine-grained Named Entity Recognition, automatically generated from Wikipedia and Wikinews. It covers 10 languages and 15 distinct entity categories.

## Task Description

- **Task Type**: Fine-grained Multilingual Named Entity Recognition (NER)
- **Input**: Wikipedia and Wikinews text
- **Output**: Identified entity spans with 15 fine-grained types
- **Domain**: General knowledge, news, encyclopedic content

## Key Features

- Large-scale automatically generated corpus
- 10 languages supported
- 15 fine-grained entity categories
- Sourced from Wikipedia and Wikinews
- Comprehensive entity type coverage

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, MYTH, PLANT, TIME, VEHI


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `multi_nerd` |
| **Dataset ID** | [extraordinarylab/multi-nerd](https://modelscope.cn/datasets/extraordinarylab/multi-nerd/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 167,993 |
| Prompt Length (Mean) | 4016.26 chars |
| Prompt Length (Min/Max) | 3915 / 4501 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "56cf0758",
      "content": "Here are some examples of named entity recognition:\n\nInput:\n2002 ging er ins Ausland und wechselte für 750.000 Pfund Sterling zu Manchester City .\n\nOutput:\n<response>2002 ging er ins Ausland und wechselte für 750.000 Pfund Sterling zu <organi ... [TRUNCATED] ...  Ensure every opening tag has a matching closing tag.\n\nText to process:\nIn der Wissenschaft und dort vor allem in der Soziologie wird der Begriff Lebensführung traditionell stark mit der religionshistorischen Arbeit von Max Weber verbunden .\n"
    }
  ],
  "target": "<response>In der Wissenschaft und dort vor allem in der Soziologie wird der Begriff Lebensführung traditionell stark mit der religionshistorischen Arbeit von <person>Max Weber</person> verbunden .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "In",
      "der",
      "Wissenschaft",
      "und",
      "dort",
      "vor",
      "allem",
      "in",
      "der",
      "Soziologie",
      "wird",
      "der",
      "Begriff",
      "Lebensführung",
      "traditionell",
      "stark",
      "mit",
      "der",
      "religionshistorischen",
      "Arbeit",
      "von",
      "Max",
      "Weber",
      "verbunden",
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
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-PER",
      "I-PER",
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
    --datasets multi_nerd \
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
    datasets=['multi_nerd'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


