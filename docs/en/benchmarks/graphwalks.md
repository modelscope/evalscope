# GraphWalks


## Overview

GraphWalks is OpenAI's long-context graph reasoning benchmark. Models receive a directed graph's edge list and return the nodes resulting from a graph operation.

## Task Description

- **Task Type**: Graph traversal and parent-node retrieval
- **Input**: The official user prompt, including three demonstrations, a directed edge list, and an operation
- **Output**: A final line containing `Final Answer: [node1, node2, ...]`
- **Domain**: Long-context reasoning over graphs

## Key Features

- All 1,150 examples from both official Parquet files: `graphwalks_128k_and_shorter.parquet` and `graphwalks_256k_to_1mil.parquet`.
- Separate `bfs` (breadth-first search) and `parents` result subsets.
- MIT-licensed data pinned to revision `be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7`, including the February 2026 parent-answer corrections and clarified BFS prompts.
- Prompts pass through unchanged. `few_shot_num=0` means no additional demonstrations; the published prompts already contain three.

## Evaluation Notes

- Use `--dataset-hub huggingface` to load the official dataset. The evaluation split is `train`; both files are loaded by the standard dataset loader. No ModelScope mirror is declared.
- Primary metric: mean per-example **F1**, with mean per-example **precision** and **recall**. Overall results weight task subsets by their sample counts, rather than pooling node counts.
- Extraction and scoring follow the [pinned OpenAI reference](https://huggingface.co/datasets/openai/graphwalks/blob/be6cc6ecf9b4d495b07d1ff53d2a16598e90fed7/README.md). Node order and duplicates do not affect scores.
- Only the literal last line is parsed, using a case-sensitive `Final Answer:` substring and a greedy bracket match. Trailing newlines fail parsing; node quotes are not removed. No response cleanup is enabled by default.
- A successfully parsed empty list against an empty target scores 1 on all metrics. Parse failures score 0, including against an empty target, and remain in the denominator.
- Real-model runs require long-context inference: the files cover nominal contexts through 1M tokens, and prompts can exceed 1.7 million characters. Character counts are not token counts. Check the model's tokenizer, chat-template overhead, input limit, and output-token headroom before running. The adapter does not truncate prompts; configure the backend to reject over-context requests and report failed or omitted samples when comparing results.
- No GPU, judge, or paid API is needed for the tests. Offline scoring and mocked end-to-end checks: `pytest tests/benchmark/test_graphwalks.py`. Full official-data verification (downloads about 319 MB on first use): `EVALSCOPE_GRAPHWALKS_DOWNLOAD=1 pytest tests/benchmark/test_graphwalks.py -k official_dataset`.
- A small live-data pipeline smoke run is `evalscope eval --datasets graphwalks --dataset-hub huggingface --model dummy --eval-type mock_llm --limit 1`. The limit applies per task subset; mock scores do not measure model quality.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `graphwalks` |
| **Dataset ID** | [openai/graphwalks](https://huggingface.co/datasets/openai/graphwalks) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `Reasoning` |
| **Metrics** | `f1`, `precision`, `recall` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,150 |
| Prompt Length (Mean) | 418581.74 chars |
| Prompt Length (Min/Max) | 2709 / 1748364 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `bfs` | 550 | 437486.04 | 2788 | 1748364 |
| `parents` | 600 | 401252.79 | 2709 | 1748284 |

## Sample Example

**Subset**: `bfs`

```json
{
  "input": [
    {
      "id": "35942846",
      "content": "\nYou will be given a graph as a list of directed edges. All nodes are at least degree 1. You will also get a description of an operation to perform on the graph.\nYour job is to execute the operation on the graph and return the set of nodes th ... [TRUNCATED 2304 chars] ... ne of your response. For example, if the operation returns the set of nodes [node1, node2, node3], your response should be:\nFinal Answer: [node1, node2, node3]\nIf the operation returns the empty set, your response should be:\nFinal Answer: []\n"
    }
  ],
  "target": "[]",
  "id": 0,
  "group_id": 0,
  "subset_key": "bfs",
  "metadata": {
    "problem_type": "bfs",
    "prompt_chars": 2650
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets graphwalks \
    --dataset-hub huggingface \
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
    datasets=['graphwalks'],
    dataset_hub='huggingface',
    dataset_args={
        'graphwalks': {
            # subset_list: ['bfs', 'parents']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
