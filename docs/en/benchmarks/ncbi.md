# NCBI


## Overview

The NCBI disease corpus is a manually annotated resource of PubMed abstracts designed for disease name recognition and normalization. It provides a gold standard for evaluating disease named entity recognition systems.

## Task Description

- **Task Type**: Disease Named Entity Recognition (NER)
- **Input**: PubMed abstract text
- **Output**: Identified disease entity spans
- **Domain**: Medical informatics, clinical NLP

## Key Features

- Manually annotated disease mentions
- PubMed abstract corpus
- Supports disease name normalization
- Gold standard for disease NER evaluation
- High-quality expert annotations

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: DISEASE (diseases, disorders, syndromes)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ncbi` |
| **Dataset ID** | [extraordinarylab/ncbi](https://modelscope.cn/datasets/extraordinarylab/ncbi/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 940 |
| Prompt Length (Mean) | 2646.25 chars |
| Prompt Length (Min/Max) | 2502 / 2988 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "e87ce89c",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nIdentification of APC2 , a homologue of the adenomatous polyposis coli tumour suppressor .\n\nOutput:\n<response>Identification of APC2 , a homologue of the <disease>adenomatous polypos ... [TRUNCATED] ...  If entity spans overlap, choose the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nClustering of missense mutations in the ataxia - telangiectasia gene in a sporadic T - cell leukaemia .\n"
    }
  ],
  "target": "<response>Clustering of missense mutations in the <disease>ataxia - telangiectasia</disease> gene in a <disease>sporadic T - cell leukaemia</disease> .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Clustering",
      "of",
      "missense",
      "mutations",
      "in",
      "the",
      "ataxia",
      "-",
      "telangiectasia",
      "gene",
      "in",
      "a",
      "sporadic",
      "T",
      "-",
      "cell",
      "leukaemia",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "O",
      "O",
      "O",
      "B-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
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
    --datasets ncbi \
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
    datasets=['ncbi'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


