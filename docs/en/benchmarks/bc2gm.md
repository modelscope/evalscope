# BC2GM


## Overview

The BC2GM (BioCreative II Gene Mention) dataset is a widely used corpus for gene mention recognition, consisting of 20,000 sentences from MEDLINE abstracts where gene and protein names have been manually annotated by domain experts.

## Task Description

- **Task Type**: Biomedical Named Entity Recognition (NER)
- **Input**: Biomedical text from MEDLINE abstracts
- **Output**: Identified gene and protein name spans
- **Domain**: Molecular biology, genetics

## Key Features

- 20,000 sentences from MEDLINE abstracts
- Expert-annotated gene and protein mentions
- Benchmark from BioCreative II challenge
- Widely used for biomedical NER evaluation
- High-quality manual annotations

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: GENE (gene and protein names)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bc2gm` |
| **Dataset ID** | [extraordinarylab/bc2gm](https://modelscope.cn/datasets/extraordinarylab/bc2gm/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 5,000 |
| Prompt Length (Mean) | 2228.52 chars |
| Prompt Length (Min/Max) | 2073 / 3129 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "06f75061",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nComparison with alkaline phosphatases and 5 - nucleotidase\n\nOutput:\n<response>Comparison with <gene>alkaline phosphatases</gene> and 5 - nucleotidase</response>\n\nInput:\nPharmacologic ... [TRUNCATED] ...  most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nPhenotypic analysis demonstrates that trio and Abl cooperate in regulating axon outgrowth in the embryonic central nervous system ( CNS ) .\n"
    }
  ],
  "target": "<response>Phenotypic analysis demonstrates that <gene>trio</gene> and <gene>Abl</gene> cooperate in regulating axon outgrowth in the embryonic central nervous system ( CNS ) .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Phenotypic",
      "analysis",
      "demonstrates",
      "that",
      "trio",
      "and",
      "Abl",
      "cooperate",
      "in",
      "regulating",
      "axon",
      "outgrowth",
      "in",
      "the",
      "embryonic",
      "central",
      "nervous",
      "system",
      "(",
      "CNS",
      ")",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "B-GENE",
      "O",
      "B-GENE",
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
    --datasets bc2gm \
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
    datasets=['bc2gm'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


