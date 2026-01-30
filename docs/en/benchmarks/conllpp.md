# CoNLL++


## Overview

The CoNLL++ dataset is a corrected and cleaner version of the test set from the widely-used CoNLL2003 NER benchmark. It provides improved annotation quality for evaluating named entity recognition systems on news text.

## Task Description

- **Task Type**: Named Entity Recognition (NER)
- **Input**: News article text
- **Output**: Identified entity spans with types
- **Domain**: News articles, general domain

## Key Features

- Corrected version of CoNLL2003 test set
- Higher annotation quality than original
- Standard NER entity types (PER, ORG, LOC, MISC)
- Widely used benchmark for NER evaluation
- Comparable results with original CoNLL2003

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC, MISC


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `conllpp` |
| **Dataset ID** | [extraordinarylab/conllpp](https://modelscope.cn/datasets/extraordinarylab/conllpp/summary) |
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
| Prompt Length (Mean) | 2732.4 chars |
| Prompt Length (Min/Max) | 2663 / 3155 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d520596a",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nEU rejects German call to boycott British lamb .\n\nOutput:\n<response><organization>EU</organization> rejects <miscellaneous>German</miscellaneous> call to boycott <miscellaneous>Briti ... [TRUNCATED] ... include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nSOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .\n"
    }
  ],
  "target": "<response>SOCCER - <location>JAPAN</location> GET LUCKY WIN , <location>CHINA</location> IN SURPRISE DEFEAT .</response>",
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
      "B-LOC",
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
    --datasets conllpp \
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
    datasets=['conllpp'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


