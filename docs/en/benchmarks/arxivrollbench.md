# ArxivRollBench

## Overview

ArxivRollBench is a rolling benchmark built from recent arXiv papers. It evaluates whether large language models can reason over fresh scientific text through three task formats: sequencing, cloze, and next-fragment prediction.

## Task Description

- **Task Type**: Multiple-choice scientific text reasoning
- **Input**: Recent arXiv text fragments with four answer choices
- **Output**: Single correct answer letter (A, B, C, or D)
- **Domains**: Computer Science, Quantitative Finance, Mathematics, Physics, Statistics, Quantitative Biology, Economics, and Electrical Engineering/System Science
- **Releases**: 2024b, 2025a, and 2026a rolling snapshots

## Key Features

- Time-aware benchmark snapshots reduce contamination-related overestimation
- Covers multiple arXiv domains and scientific writing styles
- Includes sequencing, cloze, and prediction formats under the SCP framework
- Compact `-50` split is suitable for cost-controlled API evaluation
- Full split is available as `arxivrollbench_full`

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- The default `arxivrollbench` benchmark uses compact `-50` datasets
- Use `arxivrollbench_full` for the complete public splits
- Each subset is resolved to its public Hugging Face dataset under the `liangzid` namespace
- Answers are normalized to A-D and evaluated with accuracy

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `arxivrollbench` |
| **Dataset ID** | [liangzid/arxivrollbench](https://modelscope.cn/datasets/liangzid/arxivrollbench/summary) |
| **Paper** | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/41098) |
| **Tags** | `Knowledge`, `MCQ`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**Prompt Template:**
```text
Answer the following ArxivRollBench multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets arxivrollbench \
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
    datasets=['arxivrollbench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
