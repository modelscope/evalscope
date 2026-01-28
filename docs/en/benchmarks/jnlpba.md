# JNLPBA


## Overview

The JNLPBA dataset is a widely-used resource for bio-entity recognition, consisting of 2,404 MEDLINE abstracts from the GENIA corpus annotated for five key molecular biology entity types. It is a standard benchmark for biomedical NER.

## Task Description

- **Task Type**: Biomedical Named Entity Recognition (NER)
- **Input**: Biomedical text from MEDLINE abstracts
- **Output**: Identified molecular biology entity spans
- **Domain**: Molecular biology, bioinformatics

## Key Features

- 2,404 MEDLINE abstracts from GENIA corpus
- Five molecular biology entity types
- Expert-annotated by domain specialists
- Standard benchmark for biomedical NER
- Comprehensive coverage of biomolecular entities

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: PROTEIN, DNA, RNA, CELL_LINE, CELL_TYPE


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `jnlpba` |
| **Dataset ID** | [extraordinarylab/jnlpba](https://modelscope.cn/datasets/extraordinarylab/jnlpba/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,856 |
| Prompt Length (Mean) | 3609.26 chars |
| Prompt Length (Min/Max) | 3450 / 4664 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "270411f4",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nIL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase .\n\nOutput:\n<response><dna>IL-2 gene</dna> expression and <protein>NF ... [TRUNCATED] ... ged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nNumber of glucocorticoid receptors in lymphocytes and their sensitivity to hormone action .\n"
    }
  ],
  "target": "<response>Number of <protein>glucocorticoid receptors</protein> in <cell_type>lymphocytes</cell_type> and their sensitivity to hormone action .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Number",
      "of",
      "glucocorticoid",
      "receptors",
      "in",
      "lymphocytes",
      "and",
      "their",
      "sensitivity",
      "to",
      "hormone",
      "action",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "B-PROTEIN",
      "I-PROTEIN",
      "O",
      "B-CELL_TYPE",
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
    --datasets jnlpba \
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
    datasets=['jnlpba'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


