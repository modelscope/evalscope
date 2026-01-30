# BC5CDR


## Overview

The BC5CDR corpus is a manually annotated resource of 1,500 PubMed articles developed for the BioCreative V challenge, containing over 4,400 chemical mentions, 5,800 disease mentions, and 3,100 chemical-disease interactions.

## Task Description

- **Task Type**: Biomedical Named Entity Recognition (NER)
- **Input**: PubMed article text
- **Output**: Identified chemical and disease entity spans
- **Domain**: Pharmacology, medical informatics, toxicology

## Key Features

- 1,500 PubMed articles with expert annotations
- 4,400+ chemical mentions
- 5,800+ disease mentions
- 3,100+ chemical-disease interactions
- Benchmark from BioCreative V challenge

## Evaluation Notes

- Default configuration uses **5-shot** evaluation
- Metrics: Precision, Recall, F1-Score, Accuracy
- Entity types: CHEMICAL, DISEASE


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bc5cdr` |
| **Dataset ID** | [extraordinarylab/bc5cdr](https://modelscope.cn/datasets/extraordinarylab/bc5cdr/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `NER` |
| **Metrics** | `precision`, `recall`, `f1_score`, `accuracy` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,797 |
| Prompt Length (Mean) | 3598.18 chars |
| Prompt Length (Min/Max) | 3455 / 4087 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "154a621d",
      "content": "Here are some examples of named entity recognition:\n\nInput:\nSelegiline - induced postural hypotension in Parkinson ' s disease : a longitudinal study on the effects of drug withdrawal .\n\nOutput:\n<response><chemical>Selegiline</chemical> - ind ... [TRUNCATED] ... e.\n7. Ensure every opening tag has a matching closing tag.\n\nText to process:\nTorsade de pointes ventricular tachycardia during low dose intermittent dobutamine treatment in a patient with dilated cardiomyopathy and congestive heart failure .\n"
    }
  ],
  "target": "<response><disease>Torsade de pointes ventricular tachycardia</disease> during low dose intermittent <chemical>dobutamine</chemical> treatment in a patient with <disease>dilated cardiomyopathy</disease> and <disease>congestive heart failure</disease> .</response>",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "tokens": [
      "Torsade",
      "de",
      "pointes",
      "ventricular",
      "tachycardia",
      "during",
      "low",
      "dose",
      "intermittent",
      "dobutamine",
      "treatment",
      "in",
      "a",
      "patient",
      "with",
      "dilated",
      "cardiomyopathy",
      "and",
      "congestive",
      "heart",
      "failure",
      "."
    ],
    "ner_tags": [
      "B-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "I-DISEASE",
      "O",
      "O",
      "O",
      "O",
      "B-CHEMICAL",
      "O",
      "O",
      "O",
      "O",
      "O",
      "B-DISEASE",
      "I-DISEASE",
      "O",
      "B-DISEASE",
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
    --datasets bc5cdr \
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
    datasets=['bc5cdr'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


