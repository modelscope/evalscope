# Humanity's-Last-Exam-with-Tools


## Overview

Humanity's Last Exam with Tools (`hle_tools`) is the tool-use counterpart of closed-book [`hle`](hle.md). It evaluates the same 2,500 expert-level questions from CAIS / Scale AI, but drives each sample through EvalScope's Native AgentLoop so the model can call code execution and optional web tools before answering.

This is a **distinct leaderboard row** from `hle`. Use `datasets=['hle']` for closed-book QA and `datasets=['hle_tools']` for the multi-turn tool-use protocol.

## Task Description

- **Task Type**: Expert-Level Question Answering with Tools (multi-turn AgentLoop)
- **Input**: Question with optional image (14% multimodal), plus `python_exec` (and optional MCP fetch/search)
- **Output**: Answer with explanation and confidence score
- **Domains**: Mathematics (41%), Physics (9%), Biology/Medicine (11%), Computer Science/AI (10%), Humanities (9%), Engineering (4%), Chemistry (7%), Other (9%)

## Key Features

- Reuses the official HLE dataset (`cais/hle` on ModelScope), judge (`GRADE: C/I`), subsets, and `include_multi_modal` extra param
- Default Native AgentLoop: `function_calling` strategy, `python_exec`, `local` environment, 30 steps
- Optional MCP `fetch` is attached automatically when `evalscope[mcp]` (and `mcp-server-fetch`) is installed — no paid search API key required
- Optional MCP web search (Brave, Tavily, …) can be added through `NativeAgentConfig.mcp_servers`
- Docker is recommended for isolated production runs but is **not** required; the default `local` environment is mock-friendly

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with the same HLE LLM judge
- Response format includes: Explanation, Answer, and Confidence (0-100%)
- **Note**: Set `extra_params["include_multi_modal"]` to `False` for text-only models
- Uses GRADE: C/I format for LLM judge scoring
- `datasets=['hle_tools']` enables tools by default. You do **not** need to set `TaskConfig.agent_config` unless you want to override the loop
- Passing `NativeAgentConfig` merges with defaults for any field you leave unset (`tools`, `environment`, `max_steps`, `mcp_servers`)
- `local` has no filesystem isolation; for formal runs set `environment='docker'` (see below)
- Install web fetch with `pip install evalscope[mcp]`. Paid search MCP servers are optional

## Agent Environment

Default loop (applied when `agent_config` is omitted):

- **Strategy**: `function_calling`
- **Tools**: `python_exec` (built-in). `submit` is auto-injected
- **Environment**: `local` (host subprocess). Recommended production override: `docker` + `python:3.11-slim`
- **MCP**: `mcp-server-fetch` when the optional extra is installed
- **max_steps**: 30

Override example — Docker plus fetch, and an optional search server:

```python
import sys
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['hle_tools'],
    dataset_args={
        'hle_tools': {
            'subset_list': ['Math'],
            'extra_params': {'include_multi_modal': False},
        }
    },
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        tools=['python_exec'],
        environment='docker',
        environment_extra={'sandbox_config': {'image': 'python:3.11-slim'}},
        max_steps=30,
        mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
            # Optional paid search, e.g. Brave / Tavily MCP — not required
        ],
    ),
    limit=10,
))
```


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `hle_tools` |
| **Dataset ID** | [cais/hle](https://modelscope.cn/datasets/cais/hle/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2501.14249) |
| **Tags** | `Agent`, `Knowledge`, `MultiTurn`, `QA` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,500 |
| Prompt Length (Mean) | 1029.85 chars |
| Prompt Length (Min/Max) | 234 / 21341 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Biology/Medicine` | 280 | 1259.39 | 246 | 13702 |
| `Chemistry` | 165 | 812.72 | 236 | 6942 |
| `Computer Science/AI` | 241 | 1581.02 | 263 | 11529 |
| `Engineering` | 111 | 1620.26 | 250 | 21341 |
| `Humanities/Social Science` | 219 | 1069.39 | 256 | 7028 |
| `Math` | 1,021 | 862.46 | 262 | 8952 |
| `Physics` | 230 | 1027.63 | 257 | 17139 |
| `Other` | 233 | 754.94 | 234 | 13655 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 342 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 329x12 - 14950x2780 |
| Formats | gif, jpeg, png, webp |


## Sample Example

**Subset**: `Biology/Medicine`

```json
{
  "input": [
    {
      "id": "906a518f",
      "content": "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"
    },
    {
      "id": "d03d8d4e",
      "content": [
        {
          "text": "In a bioinformatics lab, Watterson's estimator (theta) and pi (nucleotide diversity) will be calculated from variant call files which contain human phased samples with only single nucleotide variants present, and there are no completely missi ... [TRUNCATED] ... y pi (nucleotide diversity) is biased.\nC. Both Watterson's estimator (theta) and pi (nucleotide diversity) are biased.\nD. Neither Watterson's estimator (theta) nor pi (nucleotide diversity) are biased.\nE. None of the other answers are correct"
        }
      ]
    }
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "Biology/Medicine",
  "metadata": {
    "uid": "66e88728ba7d8bc0d5806f3a",
    "author_name": "Scott S",
    "rationale": "First, we recognize that all single nucleotide variants are included somewhere in the sample. It is given that, across “all samples,” there are no “missing single nucleotide variants.” Further, since “[t]he number of samples is arbitrarily la ... [TRUNCATED] ... fferent genotypes that that position, the analysis would consider these two genomes to have the same nucleotide at the position. This reduces the estimated nucleotide diversity, pi. Therefore, pi would be biased in the circumstance described.",
    "raw_subject": "Bioinformatics",
    "category": "Biology/Medicine",
    "has_image": false
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_multi_modal` | `bool` | `True` | Include multi-modal (image) questions during evaluation. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets hle_tools \
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
    datasets=['hle_tools'],
    dataset_args={
        'hle_tools': {
            # subset_list: ['Biology/Medicine', 'Chemistry', 'Computer Science/AI']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
