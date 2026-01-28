# BC4CHEMD


## Overview

The BC4CHEMD (BioCreative IV CHEMDNER) dataset is a corpus of 10,000 PubMed abstracts with 84,355 chemical entity mentions manually annotated by experts for chemical named entity recognition.

## Task Description

- **Task Type**: Chemical Named Entity Recognition (NER)
- **Input**: Scientific text from PubMed abstracts
- **Output**: Identified chemical compound name spans
- **Domain**: Chemistry, pharmacology, drug discovery

## Key Features

- 10,000 PubMed abstracts
- 84,355 chemical entity mentions
- Expert manual annotations
- Benchmark from BioCreative IV challenge
- Comprehensive chemical compound coverage

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: CHEMICAL (chemical and drug names)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bc4chemd` |
| **Dataset ID** | [extraordinarylab/bc4chemd](https://modelscope.cn/datasets/extraordinarylab/bc4chemd/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 26,364 |
| Prompt Length (Mean) | 3042.58 chars |
| Prompt Length (Min/Max) | 2879 / 3709 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "5067e19c",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nDPP6 as a candidate gene for neuroleptic - induced tardive dyskinesia .\n\nOutput:\n<response>DPP6 as a candidate gene for neuroleptic - induced tardive dyskinesia .</response>\n\nInput:\n ... [TRUNCATED] ... ry opening tag has a matching closing tag.\n\nText to process:\nEffects of docosahexaenoic acid and methylmercury on child ' s brain development due to consumption of fish by Finnish mother during pregnancy : a probabilistic modeling approach .\n"
    }
  ],
  "target": "<response>Effects of <chemical>docosahexaenoic acid</chemical> and <chemical>methylmercury</chemical> on child ' s brain development due to consumption of fish by Finnish mother during pregnancy : a probabilistic modeling approach .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Effects",
      "of",
      "docosahexaenoic",
      "acid",
      "and",
      "methylmercury",
      "on",
      "child",
      "'",
      "s",
      "brain",
      "development",
      "due",
      "to",
      "consumption",
      "of",
      "fish",
      "by",
      "Finnish",
      "mother",
      "during",
      "pregnancy",
      ":",
      "a",
      "probabilistic",
      "modeling",
      "approach",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "B-CHEMICAL",
      "I-CHEMICAL",
      "O",
      "B-CHEMICAL",
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
    --datasets bc4chemd \
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
    datasets=['bc4chemd'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


