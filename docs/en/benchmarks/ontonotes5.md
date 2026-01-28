# OntoNotes5

## Overview

OntoNotes Release 5.0 is a large, multilingual corpus containing text in English, Chinese, and Arabic across various genres. It is richly annotated with multiple layers of linguistic information including syntax, predicate-argument structure, word sense, named entities, and coreference.

## Task Description

- **Task Type**: Multi-genre Named Entity Recognition (NER)
- **Input**: Text from news, weblogs, broadcast conversations
- **Output**: Fine-grained named entity spans
- **Languages**: English, Chinese, Arabic

## Key Features

- Large-scale multilingual corpus
- Multiple genres (news, weblogs, broadcast)
- 18 fine-grained entity types
- Rich linguistic annotations
- Standard benchmark for NER evaluation

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ontonotes5` |
| **Dataset ID** | [extraordinarylab/ontonotes5](https://modelscope.cn/datasets/extraordinarylab/ontonotes5/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 8,262 |
| Prompt Length (Mean) | 3364.28 chars |
| Prompt Length (Min/Max) | 3253 / 4171 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "6215273c",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nPeople start their own businesses for many reasons .\n\nOutput:\n<response>People start their own businesses for many reasons .</response>\n\nInput:\nBut a chance to fill out sales - tax r ... [TRUNCATED] ... ening tag has a matching closing tag.\n\nText to process:\nThe following were among Friday 's offerings and pricings in the U.S. and non-U.S. capital markets , with terms and syndicate manager , as compiled by Dow Jones Capital Markets Report :\n"
    }
  ],
  "target": "<response>The following were among <date>Friday</date> 's offerings and pricings in the <geopolitical_entity>U.S.</geopolitical_entity> and <geopolitical_entity>non-U.S.</geopolitical_entity> capital markets , with terms and syndicate manager , as compiled by <organization>Dow Jones Capital Markets Report</organization> :</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "The",
      "following",
      "were",
      "among",
      "Friday",
      "'s",
      "offerings",
      "and",
      "pricings",
      "in",
      "the",
      "U.S.",
      "and",
      "non-U.S.",
      "capital",
      "markets",
      ",",
      "with",
      "terms",
      "and",
      "syndicate",
      "manager",
      ",",
      "as",
      "compiled",
      "by",
      "Dow",
      "Jones",
      "Capital",
      "Markets",
      "Report",
      ":"
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-DATE",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-GPE",
      "O",
      "B-GPE",
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
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
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
    --datasets ontonotes5 \
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
    datasets=['ontonotes5'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


