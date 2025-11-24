# Unified Evaluation with Your Index

In a nutshell: Use the mixed JSONL (generated from the sampling phase) as a single "dataset" to run evaluation, obtaining multi-level capability scores and weighted index total score in one go.

## Evaluation Configuration

::::{tab-set}
:::{tab-item} Using `TaskConfig` Object
```python
from evalscope import TaskConfig, run_task
import os

task_cfg = TaskConfig(
    model='qwen2.5-7b-instruct',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['data_collection'],
    dataset_args={
        'data_collection': {
            'local_path': 'outputs/stratified_mixed_data.jsonl',
            'shuffle': True
        }
    },
    eval_batch_size=5,
    generation_config={
        'temperature': 0.0
    }
)
run_task(task_cfg=task_cfg)
```
:::
:::{tab-item} Using CLI Command
```bash
evalscope eval \
  --model qwen2.5-7b-instruct \
  --eval-type openai_api \
  --api-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api-key "$DASHSCOPE_API_KEY" \
  --datasets data_collection \
  --dataset-args '{"data_collection":{"local_path":"outputs/stratified_mixed_data.jsonl","shuffle":true}}' \
  --eval-batch-size 5 \
  --generation-config '{"temperature":0.0}'
```
:::
::::

## Usage Conventions
- `datasets` uses `data_collection` as the fixed mixed entry point.
- `dataset_args.local_path` points to the mixed JSONL; optional `shuffle` controls sample traversal order.
- Output structure: `outputs/<timestamp>/{logs,predictions,reviews,reports,configs}`.
- Inference concurrency: `eval_batch_size`; review concurrency: `judge_worker_num`.

## Metrics & Output

The system outputs reports from five "observation perspectives":
1. **subset_level**: Subsets within original datasets (e.g., ARC-Challenge / ARC-Easy).
2. **dataset_level**: Aggregation of all subsets from the same dataset (e.g., arc).
3. **task_level**: Same task type (e.g., reasoning).
4. **tag_level**: Sample tags dimension (language, mode, etc.; Schema names can be merged into tags).
5. **category_level**: Custom grouping (e.g., reasoning_index).

**Metric Descriptions**:
- **micro_avg.**: Sum across samples then calculate overall accuracy (total correct / total samples). Subsets with more samples have greater influence.
- **macro_avg.**: Simple average of scores for "members" (subsets) at this level, each member weighted equally regardless of sample count.
- **weighted_avg.**: Weighted by defined leaf weights $w_i$ (weights are defined before sampling; sampling process doesn't change weight meaning).


**Example Output**:
```text
2025-11-24 18:44:50 - evalscope - data_collection_adapter.py - generate_report - 171 - INFO: subset_level Report:
+-----------+--------------+---------------+------------+-------+
| task_type | dataset_name |  subset_name  | micro_avg. | count |
+-----------+--------------+---------------+------------+-------+
| reasoning |     arc      | ARC-Challenge |    0.8     |   5   |
| reasoning |     arc      |   ARC-Easy    |    1.0     |   4   |
| reasoning |    ceval     |     logic     |    0.0     |   1   |
+-----------+--------------+---------------+------------+-------+
2025-11-24 18:44:50 - evalscope - data_collection_adapter.py - generate_report - 171 - INFO: dataset_level Report:
+-----------+--------------+------------+------------+---------------+-------+
| task_type | dataset_name | micro_avg. | macro_avg. | weighted_avg. | count |
+-----------+--------------+------------+------------+---------------+-------+
| reasoning |     arc      |   0.8889   |    0.9     |    0.8889     |   9   |
| reasoning |    ceval     |    0.0     |    0.0     |      0.0      |   1   |
+-----------+--------------+------------+------------+---------------+-------+
2025-11-24 18:44:50 - evalscope - data_collection_adapter.py - generate_report - 171 - INFO: task_level Report:
+-----------+------------+------------+---------------+-------+
| task_type | micro_avg. | macro_avg. | weighted_avg. | count |
+-----------+------------+------------+---------------+-------+
| reasoning |    0.8     |    0.6     |    0.3556     |  10   |
+-----------+------------+------------+---------------+-------+
2025-11-24 18:44:50 - evalscope - data_collection_adapter.py - generate_report - 171 - INFO: tag_level Report:
+------+------------+------------+---------------+-------+
| tags | micro_avg. | macro_avg. | weighted_avg. | count |
+------+------------+------------+---------------+-------+
|  en  |   0.8889   |    0.9     |    0.8889     |   9   |
|  zh  |    0.0     |    0.0     |      0.0      |   1   |
+------+------------+------------+---------------+-------+
2025-11-24 18:44:50 - evalscope - data_collection_adapter.py - generate_report - 171 - INFO: category_level Report:
+-----------------+------------+------------+---------------+-------+
|    category0    | micro_avg. | macro_avg. | weighted_avg. | count |
+-----------------+------------+------------+---------------+-------+
| reasoning_index |    0.8     |    0.6     |    0.3556     |  10   |
+-----------------+------------+------------+---------------+-------+
```

## Common Issues
- **Index score doesn't match expectations**: Expand Schema to verify normalized weights; confirm you're not misunderstanding weight effects with uniform sampling.
- **Changes only require re-running**: If JSONL content unchanged, enable `--use-cache` to save inference time.
- **Slow evaluation with large datasets**: Increase `eval_batch_size`; ensure model/API supports concurrent requests.

## Next Steps
- **Visualization**: `evalscope app` opens an interactive interface to filter dimensions and models.
- **Versioning**: Generate multiple JSONLs for different weight configurations, named `weighted_v1.jsonl` / `weighted_v2.jsonl` and evaluate separately.