# TableBench


## Overview

TableBench evaluates table question answering in realistic scenarios: 886 questions over real tables in
18 subcategories of four types (fact checking, numerical reasoning, data analysis and visualization).
This adapter covers the 836 text-answer questions of the first three types.

## Task Description

- **Task Type**: Table question answering
- **Input**: The official instruction with the table in JSON format and the question
- **Output**: Reasoning (depending on the instruction type), then `Final Answer: ...`
- **Domain**: Tables from finance, sports, demographics, science and more

## Key Features

- Fact checking (96), numerical reasoning (397: aggregation, arithmetic, comparison, counting, ranking,
  multi-hop, time-based, domain-specific) and data analysis (343: correlation, trend forecasting,
  statistical, impact, anomaly, causal, descriptive analysis)
- Uses the cleaned 2025-04-18 release of the test set and the refined official metrics
- Three official prompting styles: `TCoT` (textual chain-of-thought, default), `SCoT` (symbolic CoT)
  and `DP` (direct prompting)

## Evaluation Notes

- `mix_metric` (primary) is the official Mix_Metric: each question is scored with the metric the official
  script assigns to its subtype, then the scores are averaged:
  - EM for fact checking, numerical reasoning and impact analysis
  - EM with 10% numeric tolerance for correlation analysis, trend forecasting and statistical analysis
  - ROUGE-L for the other data-analysis subtypes
- The answer is the text after the first `Final Answer: ` on that line, as in the official parser
- As in the official code, prediction and reference are normalized (lowercase, punctuation and articles
  removed) before scoring, so decimal points, signs and percent signs are not compared
- Choose the prompting style with `extra_params={'instruction_type': 'DP'}`
- Visualization (50 chart-generation questions) needs execution of model-written plotting code and is
  not included yet
- [Paper](https://arxiv.org/abs/2408.09174) | [Code](https://github.com/TableBench/TableBench) |
  [Dataset](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench) |
  [Leaderboard](https://tablebench.github.io/)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `tablebench` |
| **Dataset ID** | [Multilingual-Multimodal-NLP/TableBench](https://modelscope.cn/datasets/Multilingual-Multimodal-NLP/TableBench/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2408.09174) |
| **Tags** | `QA`, `Reasoning` |
| **Metrics** | `mix_metric` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction_type` | `str` | `TCoT` | Official prompting style: TCoT (textual chain-of-thought), SCoT (symbolic CoT) or DP (direct answer). Choices: ['TCoT', 'SCoT', 'DP'] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tablebench \
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
    datasets=['tablebench'],
    dataset_args={
        'tablebench': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
