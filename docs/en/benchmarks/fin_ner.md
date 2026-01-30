# FinNER


## Overview

The FinNER dataset is a corpus of financial agreements from public U.S. Security and Exchange Commission (SEC) filings, annotated with Person, Organization, Location, and Miscellaneous entities to support information extraction for credit risk assessment.

## Task Description

- **Task Type**: Financial Named Entity Recognition (NER)
- **Input**: Financial agreement text from SEC filings
- **Output**: Identified entity spans with types
- **Domain**: Finance, legal documents, credit risk

## Key Features

- Financial agreements from SEC filings
- Annotated for credit risk assessment applications
- Standard NER entity types adapted for finance
- Specialized for financial document processing
- Useful for legal and financial AI applications

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PER, ORG, LOC, MISC


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `fin_ner` |
| **Dataset ID** | [extraordinarylab/fin-ner](https://modelscope.cn/datasets/extraordinarylab/fin-ner/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 305 |
| Prompt Length (Mean) | 2891.13 chars |
| Prompt Length (Min/Max) | 2663 / 6149 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d8a8baf8",
      "content": "Here are some examples of named entity recognition:\n\nInput:\n( l ) \" Tranche A Shares \" has the meaning as defined in the Subscription Agreement .\n\nOutput:\n<response>( l ) \" Tranche A Shares \" has the meaning as defined in the Subscription Agr ... [TRUNCATED] ... osing tag.\n\nText to process:\nSubordinated Loan Agreement - Silicium de Provence SAS and Evergreen Solar Inc . 7 - December 2007 [ HERBERT SMITH LOGO ] ................................ 2007 SILICIUM DE PROVENCE SAS and EVERGREEN SOLAR , INC .\n"
    }
  ],
  "target": "<response>Subordinated Loan Agreement - <organization>Silicium de Provence SAS</organization> and <organization>Evergreen Solar Inc</organization> . 7 - December 2007 [ <person>HERBERT SMITH</person> LOGO ] ................................ 2007 <organization>SILICIUM DE PROVENCE SAS</organization> and <organization>EVERGREEN SOLAR</organization> , INC .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Subordinated",
      "Loan",
      "Agreement",
      "-",
      "Silicium",
      "de",
      "Provence",
      "SAS",
      "and",
      "Evergreen",
      "Solar",
      "Inc",
      ".",
      "7",
      "-",
      "December",
      "2007",
      "[",
      "HERBERT",
      "SMITH",
      "LOGO",
      "]",
      "................................",
      "2007",
      "SILICIUM",
      "DE",
      "PROVENCE",
      "SAS",
      "and",
      "EVERGREEN",
      "SOLAR",
      ",",
      "INC",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
      "O",
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-PER",
      "I-PER",
      "O",
      "O",
      "O",
      "O",
      "B-ORG",
      "I-ORG",
      "I-ORG",
      "I-ORG",
      "O",
      "B-ORG",
      "I-ORG",
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
    --datasets fin_ner \
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
    datasets=['fin_ner'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


