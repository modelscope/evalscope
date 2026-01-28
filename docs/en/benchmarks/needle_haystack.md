# Needle-in-a-Haystack


## Overview

Needle in a Haystack is a benchmark focused on evaluating information retrieval capabilities in long-context scenarios. It tests a model's ability to find specific information (needles) within large documents (haystacks).

## Task Description

- **Task Type**: Long-Context Information Retrieval
- **Input**: Long document with embedded target information + retrieval question
- **Output**: Extracted target information (needle)
- **Domains**: Long-context understanding, information retrieval

## Key Features

- Tests retrieval across varying context lengths (1K-32K+ tokens)
- Tests retrieval at different document depths (0%-100%)
- Supports both English and Chinese corpora
- Generates synthetic samples with configurable parameters
- Produces heatmap visualizations of performance

## Evaluation Notes

- Default context lengths: **1,000 to 32,000** tokens (configurable)
- Default depth percentages: **0% to 100%** (configurable)
- Primary metric: **Accuracy** on retrieval
- Uses LLM judge for flexible answer matching
- Configurable via extra_params: needles, context lengths, depth intervals, tokenizer
- [Usage Example](https://evalscope.readthedocs.io/en/latest/third_party/needle_haystack.html)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `needle_haystack` |
| **Dataset ID** | [AI-ModelScope/Needle-in-a-Haystack-Corpus](https://modelscope.cn/datasets/AI-ModelScope/Needle-in-a-Haystack-Corpus/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `Retrieval` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 200 |
| Prompt Length (Mean) | 45063.55 chars |
| Prompt Length (Min/Max) | 1361 / 137407 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `english` | 100 | 70387.8 | 3898 | 137407 |
| `chinese` | 100 | 19739.3 | 1361 | 37893 |

## Sample Example

**Subset**: `english`

```json
{
  "input": [
    {
      "id": "f617334d",
      "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
    },
    {
      "id": "a832dc68",
      "content": "Please read the following text and answer the question below.\n\n<text>\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n\nWant to start a startup?  Get funded by\nY Combinator.\n\n\n\n\nJuly 2004(This  ... [TRUNCATED] ... ow do you\nget them to come and work for you?  And then of course there's the\nquestion, how do\n</text>\n\n<question>\nWhat is the best thing to do in San Francisco?\n</question>\n\nDon't give information outside the document or repeat your findings."
    }
  ],
  "target": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "context": "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n\nWant to start a startup?  Get funded by\nY Combinator.\n\n\n\n\nJuly 2004(This essay is derived from a talk at Oscon 2004.)\nA few months ago I finish ... [TRUNCATED] ... m, we need to understand these\nespecially productive people.  What motivates them?  What do they\nneed to do their jobs?  How do you recognize them? How do you\nget them to come and work for you?  And then of course there's the\nquestion, how do",
    "context_length": 1000,
    "depth_percent": 0
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**System Prompt:**
```text
You are a helpful AI bot that answers questions for a user. Keep your response short and direct
```

**Prompt Template:**
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

<question>
{question}
</question>

Don't give information outside the document or repeat your findings.
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retrieval_question` | `str` | `What is the best thing to do in San Francisco?` | Question used for retrieval evaluation. |
| `needles` | `list[str]` | `['\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n']` | List of factual needle strings inserted into the context. |
| `context_lengths_min` | `int` | `1000` | Minimum context length (tokens) to generate synthetic samples. |
| `context_lengths_max` | `int` | `32000` | Maximum context length (tokens) to generate synthetic samples. |
| `context_lengths_num_intervals` | `int` | `10` | Number of intervals between min and max context lengths. |
| `document_depth_percent_min` | `int` | `0` | Minimum insertion depth percentage for needles. |
| `document_depth_percent_max` | `int` | `100` | Maximum insertion depth percentage for needles. |
| `document_depth_percent_intervals` | `int` | `10` | Number of intervals between min and max depth percentages. |
| `tokenizer_path` | `str` | `Qwen/Qwen3-0.6B` | Tokenizer checkpoint path used for tokenization. |
| `show_score` | `bool` | `False` | Render numerical scores on heatmap output images. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets needle_haystack \
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
    datasets=['needle_haystack'],
    dataset_args={
        'needle_haystack': {
            # subset_list: ['english', 'chinese']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


