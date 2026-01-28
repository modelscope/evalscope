# GeniaNER

## Overview

GeniaNER is a large-scale biomedical NER dataset consisting of 2,000 MEDLINE abstracts with over 400,000 words and almost 100,000 annotations for biological terms. It is one of the most comprehensive resources for biomedical entity recognition.

## Task Description

- **Task Type**: Biomedical Named Entity Recognition (NER)
- **Input**: MEDLINE abstracts from GENIA corpus
- **Output**: Identified biological entity spans
- **Domain**: Molecular biology, bioinformatics

## Key Features

- 2,000 MEDLINE abstracts
- Over 400,000 words
- Nearly 100,000 biological term annotations
- Five molecular biology entity types
- Comprehensive coverage of biomolecular entities

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: CELL_LINE, CELL_TYPE, DNA, PROTEIN, RNA

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `genia_ner` |
| **Dataset ID** | [extraordinarylab/genia-ner](https://modelscope.cn/datasets/extraordinarylab/genia-ner/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,854 |
| Prompt Length (Mean) | 3921.68 chars |
| Prompt Length (Min/Max) | 3781 / 4433 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "3bb926a4",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nIL-2 gene expression and NF-kappa B activation through CD28 requires reactive oxygen production by 5-lipoxygenase .\n\nOutput:\n<response><dna>IL-2 gene</dna> expression and <protein>NF ... [TRUNCATED] ...  a matching closing tag.\n\nText to process:\nThere is a single methionine codon-initiated open reading frame of 1,458 nt in frame with a homeobox and a CAX repeat , and the open reading frame is predicted to encode a protein of 51,659 daltons.\n"
    }
  ],
  "target": "<response>There is a single <dna>methionine codon-initiated open reading frame</dna> of 1,458 nt in frame with a <dna>homeobox</dna> and a <dna>CAX repeat</dna> , and the <dna>open reading frame</dna> is predicted to encode a protein of 51,659 daltons.</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "There",
      "is",
      "a",
      "single",
      "methionine",
      "codon-initiated",
      "open",
      "reading",
      "frame",
      "of",
      "1,458",
      "nt",
      "in",
      "frame",
      "with",
      "a",
      "homeobox",
      "and",
      "a",
      "CAX",
      "repeat",
      ",",
      "and",
      "the",
      "open",
      "reading",
      "frame",
      "is",
      "predicted",
      "to",
      "encode",
      "a",
      "protein",
      "of",
      "51,659",
      "daltons."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-DNA",
      "I-DNA",
      "I-DNA",
      "I-DNA",
      "I-DNA",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-DNA",
      "O",
      "O",
      "B-DNA",
      "I-DNA",
      "O",
      "O",
      "O",
      "B-DNA",
      "I-DNA",
      "I-DNA",
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
    --datasets genia_ner \
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
    datasets=['genia_ner'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


