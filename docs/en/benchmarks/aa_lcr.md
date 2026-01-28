# AA-LCR


## Overview

AA-LCR (Artificial Analysis Long Context Retrieval) is a benchmark for evaluating long-context retrieval and reasoning capabilities of language models. It requires models to find and synthesize information across multiple documents.

## Task Description

- **Task Type**: Long-Context Question Answering
- **Input**: Multiple documents + question requiring cross-document reasoning
- **Output**: Answer synthesized from document information
- **Context**: Very long context (multiple documents concatenated)

## Key Features

- Tests long-context retrieval abilities
- Multiple document understanding
- Cross-document reasoning required
- LLM-based judging for answer correctness
- Auto-download of document corpus

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy** (via LLM judge)
- Evaluates on **test** split
- Documents auto-downloaded if `text_dir` not specified
- Judge prompt compares candidate answer against reference


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `aa_lcr` |
| **Dataset ID** | [evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `LongContext`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 100 |
| Prompt Length (Mean) | 414674.06 chars |
| Prompt Length (Min/Max) | 240709 / 548771 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "0b26b81d",
      "content": "\nBEGIN INPUT DOCUMENTS\n\nBEGIN DOCUMENT 1:\n[Contents lists available at ScienceDirect](https://www.elsevier.com/locate/techfore)\n\n# Technological Forecasting & Social Change\n\n[journal homepage: www.elsevier.com/locate/techfore](http://www.else ... [TRUNCATED] ...  and undertakings issued by the ACCC. Identify and rank the industries explicitly mentioned in the paragraphs, according to the number of infringements over the past three decades. Exclude Broadcasting Industry from the answer.\n\nEND QUESTION\n"
    }
  ],
  "target": "1. Airline Industry (12)\\n2. Accommodation Industry (4)",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question": "Based on the provided documents, there appears to be a correlation between industry concentration and the frequency of consumer-related infringements and undertakings issued by the ACCC. Identify and rank the industries explicitly mentioned in the paragraphs, according to the number of infringements over the past three decades. Exclude Broadcasting Industry from the answer.",
    "data_source_urls": "https://competition-policy.ec.europa.eu/system/files/2024-06/A_taxonomy_of_industry_competition_launch.pdf;https://www.industry.gov.au/sites/default/files/2023-11/barriers-to-collaboration-and-commercialisation-iisa.pdf;https://e61.in/wp-content/uploads/2023/08/The-State-of-Competition.pdf;https://uu.diva-portal.org/smash/get/diva2:1798138/FULLTEXT01.pdf;https://one.oecd.org/document/DAF/COMP(2023)14/en/pdf",
    "input_tokens": 94494
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text

BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION

```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_dir` | `str | null` | `None` | Local directory containing extracted AA-LCR text files; if null will auto-download & extract. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets aa_lcr \
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
    datasets=['aa_lcr'],
    dataset_args={
        'aa_lcr': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


