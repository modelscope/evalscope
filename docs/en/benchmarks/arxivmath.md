# ArXiv-Math

## Overview

ArXiv-Math is a benchmark of 103 research-level mathematics problems extracted from arXiv preprints. These problems represent cutting-edge mathematical research and test the ability of language models to reason about advanced mathematical concepts at the frontier of knowledge.

## Task Description

- **Task Type**: Research-Level Mathematics Problem Solving
- **Input**: Advanced mathematical problem from arXiv papers
- **Output**: Step-by-step solution with final answer
- **Difficulty**: Research / graduate level

## Key Features

- 103 problems sourced from arXiv preprints (December 2024 - March 2025)
- Four monthly subsets: december, february, january, march
- Covers diverse areas: algebra, combinatorics, analysis, geometry, number theory
- Problems require deep mathematical reasoning and domain expertise
- Represents the frontier of mathematical research difficulty

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should be formatted within `\boxed{}` for proper extraction
- Numeric accuracy metric with symbolic equivalence checking
- Results can be broken down by monthly competition subset

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `arxivmath` |
| **Dataset ID** | [evalscope/arxivmath](https://modelscope.cn/datasets/evalscope/arxivmath/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 103 |
| Prompt Length (Mean) | 622.88 chars |
| Prompt Length (Min/Max) | 224 / 1392 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `arxiv/december` | 17 | 720.88 | 256 | 1392 |
| `arxiv/february` | 32 | 573.78 | 269 | 1147 |
| `arxiv/january` | 23 | 711.17 | 325 | 1270 |
| `arxiv/march` | 31 | 554.32 | 224 | 1213 |

## Sample Example

**Subset**: `arxiv/december`

```json
{
  "input": [
    {
      "id": "c7cbf85d",
      "content": "Problem:\nLet $k$ be a field, let $V$ be a $k$-vector space of dimension $d$, and let $G\\subseteq GL(V)$ be a finite group. Set $r:=\\dim_k (V^*)^G$ and assume $r\\ge 1$. Let $R:=k[V]^G$ be the invariant ring, and write its Hilbert quasi-polynom ... [TRUNCATED 71 chars] ... {d-2}+\\cdots+a_1(n)n+a_0(n),\n\\]\nwhere each $a_i(n)$ is a periodic function of $n$. Compute the sum of the indices $i\\in\\{0,1,\\dots,d-1\\}$ for which $a_i(n)$ is constant.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
    }
  ],
  "target": "\\frac{r(2d-r-1)}{2}",
  "id": 0,
  "group_id": 0,
  "subset_key": "arxiv/december",
  "metadata": {
    "problem_idx": 1,
    "problem_type": [
      ""
    ],
    "source": 2512.00811
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Problem:
{question}

Please reason step by step, and put your final answer within \boxed{{}}.

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets arxivmath \
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
    datasets=['arxivmath'],
    dataset_args={
        'arxivmath': {
            # subset_list: ['arxiv/december', 'arxiv/february', 'arxiv/january']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
