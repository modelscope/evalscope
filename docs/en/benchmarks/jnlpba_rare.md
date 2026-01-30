# JNLPBA-Rare


## Overview

The JNLPBA-Rare dataset is a specialized subset of the JNLPBA test set created to evaluate zero-shot performance on its least frequent entity types: RNA and cell line. It tests model ability to recognize rare biomedical entities.

## Task Description

- **Task Type**: Rare Biomedical Named Entity Recognition (NER)
- **Input**: Biomedical text from MEDLINE abstracts
- **Output**: Identified RNA and cell line entity spans
- **Domain**: Molecular biology, bioinformatics

## Key Features

- Focuses on rare entity types (RNA, cell line)
- Subset of JNLPBA for zero-shot evaluation
- Tests handling of infrequent biomedical entities
- Challenging benchmark for entity recognition
- Useful for evaluating long-tail performance

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: RNA, CELL_LINE


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `jnlpba_rare` |
| **Dataset ID** | [extraordinarylab/jnlpba-rare](https://modelscope.cn/datasets/extraordinarylab/jnlpba-rare/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 465 |
| Prompt Length (Mean) | 1111.66 chars |
| Prompt Length (Min/Max) | 976 / 1403 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "11496ce5",
      "content": "You are a named entity recognition system that identifies the following entity types:\nrna (Names of RNA molecules), cell_line (Names of specific, cultured cell lines)\n\nProcess the provided text and mark all named entities with XML-style tags. ... [TRUNCATED] ... rlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nOctamer-binding proteins from B or HeLa cells stimulate transcription of the immunoglobulin heavy-chain promoter in vitro .\n"
    }
  ],
  "target": "<response>Octamer-binding proteins from <cell_line>B or HeLa cells</cell_line> stimulate transcription of the immunoglobulin heavy-chain promoter in vitro .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Octamer-binding",
      "proteins",
      "from",
      "B",
      "or",
      "HeLa",
      "cells",
      "stimulate",
      "transcription",
      "of",
      "the",
      "immunoglobulin",
      "heavy-chain",
      "promoter",
      "in",
      "vitro",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "B-CELL_LINE",
      "I-CELL_LINE",
      "I-CELL_LINE",
      "I-CELL_LINE",
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
    --datasets jnlpba_rare \
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
    datasets=['jnlpba_rare'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


