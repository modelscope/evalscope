# CrossNER

## Overview

CrossNER is a fully-labeled collection of named entity recognition (NER) data spanning over five diverse domains: AI, Literature, Music, Politics, and Science. It enables cross-domain NER evaluation and domain adaptation research.

## Task Description

- **Task Type**: Cross-Domain Named Entity Recognition (NER)
- **Input**: Text from five specialized domains
- **Output**: Domain-specific entity spans
- **Domain**: AI, Literature, Music, Politics, Science

## Key Features

- Five diverse domain subsets
- Domain-specific entity types per subset
- Enables cross-domain transfer evaluation
- Fully labeled with expert annotations
- Useful for domain adaptation research

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Subsets: ai, literature, music, politics, science
- Entity types vary by domain subset

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cross_ner` |
| **Dataset ID** | [extraordinarylab/cross-ner](https://modelscope.cn/datasets/extraordinarylab/cross-ner/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,506 |
| Prompt Length (Mean) | 5687.97 chars |
| Prompt Length (Min/Max) | 5407 / 6007 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `ai` | 431 | 5562.3 | 5407 | 5878 |
| `literature` | 416 | 5725.13 | 5566 | 6007 |
| `music` | 465 | 5737.0 | 5570 | 5962 |
| `politics` | 651 | 5701.8 | 5527 | 5935 |
| `science` | 543 | 5700.71 | 5548 | 5995 |

## Sample Example

**Subset**: `ai`

```json
{
  "input": [
    {
      "id": "3a78cbcf",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nPopular approaches of opinion-based recommender system utilize various techniques including text mining , information retrieval , sentiment analysis ( see also Multimodal sentiment a ... [TRUNCATED] ...  the most specific entity type.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nTypical generative model approaches include naive Bayes classifier s , Gaussian mixture model s , variational autoencoders and others .\n"
    }
  ],
  "target": "<response>Typical generative model approaches include <algorithm>naive Bayes classifier</algorithm> s , <algorithm>Gaussian mixture model</algorithm> s , <algorithm>variational autoencoders</algorithm> and others .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Typical",
      "generative",
      "model",
      "approaches",
      "include",
      "naive",
      "Bayes",
      "classifier",
      "s",
      ",",
      "Gaussian",
      "mixture",
      "model",
      "s",
      ",",
      "variational",
      "autoencoders",
      "and",
      "others",
      "."
    ],
    "ner_tags": [
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-ALGORITHM",
      "I-ALGORITHM",
      "I-ALGORITHM",
      "O",
      "O",
      "B-ALGORITHM",
      "I-ALGORITHM",
      "I-ALGORITHM",
      "O",
      "O",
      "B-ALGORITHM",
      "I-ALGORITHM",
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
    --datasets cross_ner \
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
    datasets=['cross_ner'],
    dataset_args={
        'cross_ner': {
            # subset_list: ['ai', 'literature', 'music']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


