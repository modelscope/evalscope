# AnatEM


## Overview

The AnatEM corpus is an extensive resource for anatomical entity recognition, created by extending and combining previous corpora. It includes over 13,000 annotations across 1,212 biomedical documents, focusing on identifying anatomical structures from subcellular components to organ systems.

## Task Description

- **Task Type**: Biomedical Named Entity Recognition (NER)
- **Input**: Biomedical text from PubMed abstracts
- **Output**: Identified anatomical entity spans with types
- **Domain**: Anatomy, biomedical literature

## Key Features

- Over 13,000 anatomical entity annotations
- 1,212 biomedical documents from PubMed
- Comprehensive anatomical coverage (cells to organs)
- Manual expert annotation
- Useful for biomedical text mining

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: ANATOMY (subcellular structures, cells, tissues, organs)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `anat_em` |
| **Dataset ID** | [extraordinarylab/anat-em](https://modelscope.cn/datasets/extraordinarylab/anat-em/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 3,830 |
| Prompt Length (Mean) | 3007.08 chars |
| Prompt Length (Min/Max) | 2861 / 3652 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "64b20a23",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nImmunostaining and confocal analysis\n\nOutput:\n<response>Immunostaining and confocal analysis</response>\n\nInput:\nDNA labelling and staining with 5 - bromo - 2 ' - deoxyuridine ( BrdU  ... [TRUNCATED] ... Do not include explanations, just the tagged text.\n6. If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\n( a ) Schematic drawing of the magnetic tweezers .\n"
    }
  ],
  "target": "<response>( a ) Schematic drawing of the magnetic tweezers .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "(",
      "a",
      ")",
      "Schematic",
      "drawing",
      "of",
      "the",
      "magnetic",
      "tweezers",
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
    --datasets anat_em \
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
    datasets=['anat_em'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


