# BrowseComp


## Overview

BrowseComp is an OpenAI benchmark for evaluating browsing and search agents. It contains 1,266 hard-to-find, fact-seeking questions with short, verifiable answers.

## Task Description

- **Task Type**: Search-agent factual question answering
- **Input**: Challenging natural-language question that generally requires persistent web browsing
- **Output**: Explanation, exact answer, and confidence
- **Grading**: LLM judge compares the final answer against the reference answer

## Key Features

- Tests persistence, creative search, and multi-hop evidence gathering
- Uses short answers to keep grading tractable
- Official data is distributed as encrypted CSV rows and decrypted at evaluation time
- Compatible with EvalScope agent configuration via the default agent loop path

## Evaluation Notes

- Default evaluation uses the official public BrowseComp CSV.
- Use `dataset_args={'browsecomp': {'local_path': '/path/to/browse_comp_test_set.csv'}}` to evaluate from a local copy.
- The primary metric is `is_correct`; `is_incorrect` is also reported.
- LLM judge is enabled by default. `JudgeStrategy.RULE` falls back to normalized exact match.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `browsecomp` |
| **Dataset ID** | [browse_comp_test_set.csv](https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv) |
| **Paper** | [Paper](https://arxiv.org/abs/2504.12516) |
| **Tags** | `Agent`, `Knowledge`, `QA` |
| **Metrics** | `is_correct`, `is_incorrect` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,266 |
| Prompt Length (Mean) | 811.02 chars |
| Prompt Length (Min/Max) | 424 / 2219 chars |

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets browsecomp \
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
    datasets=['browsecomp'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


