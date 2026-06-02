# Custom Text Retrieval Evaluation Dataset

## Overview

When you need to evaluate the retrieval capabilities of Embedding models on domain-specific data (e.g., legal, medical, financial) or private datasets, you can build a custom dataset for evaluation. EvalScope, built on the MTEB framework, supports defining corpus, queries, and relevance judgments through simple JSONL files for end-to-end retrieval evaluation.

## Step 1: Prepare Data

### Directory Structure

Create a directory containing the following three JSONL files:

```
my_retrieval_data/
├── corpus.jsonl
├── queries.jsonl
└── qrels.jsonl
```

### corpus.jsonl

The corpus file, with one JSON object per line representing a document to be retrieved.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_id` | str | Yes | Unique document identifier |
| `text` | str | Yes | Document body content |
| `title` | str | No | Document title, used to assist retrieval |

```json
{"_id": "doc1", "text": "Climate change is leading to more extreme weather patterns.", "title": "Climate Change"}
{"_id": "doc2", "text": "The stock market surged today, led by tech stocks."}
{"_id": "doc3", "text": "Artificial intelligence is transforming industries by automating tasks and providing insights."}
```

### queries.jsonl

The queries file, with one JSON object per line representing a retrieval query.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `_id` | str | Yes | Unique query identifier |
| `text` | str | Yes | Query text |

```json
{"_id": "query1", "text": "What are the impacts of climate change?"}
{"_id": "query2", "text": "What caused the stock market to rise today?"}
{"_id": "query3", "text": "How is artificial intelligence changing industries?"}
```

### qrels.jsonl

The relevance judgments file, with one JSON object per line defining the relevance between a query and a document.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query-id` | str | Yes | Corresponds to `_id` in queries.jsonl |
| `corpus-id` | str | Yes | Corresponds to `_id` in corpus.jsonl |
| `score` | int/float | Yes | Relevance score |

```json
{"query-id": "query1", "corpus-id": "doc1", "score": 1}
{"query-id": "query2", "corpus-id": "doc2", "score": 1}
{"query-id": "query3", "corpus-id": "doc3", "score": 1}
```

**Score semantics:**

- **Binary (0/1)**: `1` means relevant, `0` means not relevant. Suitable for most retrieval scenarios.
- **Graded relevance**: Use multi-level scores (e.g., 0, 1, 2, 3), where higher values indicate greater relevance. Suitable for fine-grained evaluation that distinguishes "partially relevant" from "highly relevant".

### Data Quality Recommendations

```{tip}
- The corpus should contain at least **100** documents and at least **50** queries for statistically meaningful evaluation results.
- Ensure each query has at least **1** relevant document annotation.
- When using binary labels, consider also annotating negative samples (score=0) to more accurately assess the model's discrimination ability.
- Include enough irrelevant "distractor" documents in the corpus to simulate real retrieval scenarios.
```

## Step 2: Configure and Run Evaluation

### Complete Configuration Example

```python
from evalscope.run import run_task

task_cfg = {
    "work_dir": "outputs",
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "pooling_mode": None,
                "max_seq_length": 512,
                "prompt": "",
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {
                    "batch_size": 128,
                },
            }
        ],
        "eval": {
            "custom_tasks": [
                {
                    "name": "CustomRetrieval",
                    "data_path": "my_retrieval_data",
                }
            ],
            "overwrite_results": True,
            "limits": 500,
        },
    },
}

run_task(task_cfg=task_cfg)
```

### Run Command

Save the above code as `run_eval.py`, then execute:

```bash
python run_eval.py
```

### Interpreting Results

After evaluation completes, results are saved in the `outputs/` directory. Key metrics include:

- **NDCG@k**: Normalized Discounted Cumulative Gain, measuring ranking quality (k=1,3,5,10)
- **MAP@k**: Mean Average Precision
- **Recall@k**: Recall rate
- **Precision@k**: Precision rate

Scores range from 0 to 1, with higher values indicating better retrieval performance.

## Parameter Reference

Each task in the `custom_tasks` list corresponds to a `CustomTaskConfig` with the following fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `"CustomRetrieval"` | Task name, used to identify evaluation results |
| `data_path` | str | (required) | Dataset directory path, must contain `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl` |
| `eval_splits` | List[str] | `["test"]` | Evaluation splits list |

```{note}
Other evaluation parameters (such as `models`, `limits`, `overwrite_results`, etc.) are consistent with the default configuration. See [MTEB Evaluation Parameter Reference](../../user_guides/backend/rageval_backend/mteb.md#parameter-explanation) for details.
```

## FAQ

### Field Format Errors

- Data files must be in **JSONL** format (one independent JSON object per line), not standard JSON array format.
- The `_id` field value must be a **string** type. Even numeric IDs should be written as `"123"` rather than `123`.
- Field names in `qrels.jsonl` use hyphens: `query-id` and `corpus-id`, not underscores.

### Impact of Insufficient Data

- When the corpus has too few documents (e.g., fewer than 10), metrics like Recall@10 may always be 1.0, making it impossible to effectively differentiate between models.
- Too few queries lead to high variance in evaluation results, lacking statistical significance.
- It is recommended to prepare at least hundreds of samples based on your actual business scenario.

### Converting from Other Formats

If you already have data in TSV or CSV format, you can convert it to JSONL as follows:

```python
import csv
import json

# Example: TSV/CSV to corpus.jsonl
with open("corpus.tsv", "r") as fin, open("corpus.jsonl", "w") as fout:
    reader = csv.DictReader(fin, delimiter="\t")  # For CSV, use delimiter=","
    for row in reader:
        obj = {"_id": row["id"], "text": row["text"]}
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
```
