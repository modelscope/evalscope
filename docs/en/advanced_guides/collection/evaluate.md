# Unified Evaluation

After obtaining the sampled data, you can proceed with the unified evaluation.

## Evaluation Configuration

Configure the evaluation task, for example:

```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen2.5',
    api_url='http://127.0.0.1:8801/v1',
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,
    datasets=['data_collection'],
    dataset_args={'data_collection': {
        'dataset_id': 'outputs/mixed_data.jsonl'
    }},
)
run_task(task_cfg=task_cfg)
```

It is important to note that:
- The dataset name specified in `datasets` is fixed as `data_collection`, indicating the evaluation of the mixed dataset.
- In `dataset_args`, you need to specify `dataset_id`, which indicates the local path to the evaluation dataset or the dataset ID on ModelScope.

## Evaluation Results

The evaluation results are saved by default in the `outputs/` directory, containing reports with four levels:

- `subset_level`: Average scores and counts for each subset.
- `dataset_level`: Average scores and counts for each dataset.
- `task_level`: Average scores and counts for each task.
- `tag_level`: Average scores and counts for each tag, with the schema name also included as a tag in the `tags` column.

For example, the evaluation results might look like this:

```text
2024-12-30 20:03:54,582 - evalscope - INFO - subset_level Report:
+-----------+------------------+---------------+---------------+-------+
| task_type |   dataset_name   |  subset_name  | average_score | count |
+-----------+------------------+---------------+---------------+-------+
|   math    | competition_math |    default    |      0.0      |  38   |
| reasoning |       race       |     high      |    0.3704     |  27   |
| reasoning |       race       |    middle     |      0.5      |  12   |
| reasoning |       arc        |   ARC-Easy    |    0.5833     |  12   |
|   math    |      gsm8k       |     main      |    0.1667     |   6   |
| reasoning |       arc        | ARC-Challenge |      0.4      |   5   |
+-----------+------------------+---------------+---------------+-------+
2024-12-30 20:03:54,582 - evalscope - INFO - dataset_level Report:
+-----------+------------------+---------------+-------+
| task_type |   dataset_name   | average_score | count |
+-----------+------------------+---------------+-------+
| reasoning |       race       |    0.4103     |  39   |
|   math    | competition_math |      0.0      |  38   |
| reasoning |       arc        |    0.5294     |  17   |
|   math    |      gsm8k       |    0.1667     |   6   |
+-----------+------------------+---------------+-------+
2024-12-30 20:03:54,582 - evalscope - INFO - task_level Report:
+-----------+---------------+-------+
| task_type | average_score | count |
+-----------+---------------+-------+
| reasoning |    0.4464     |  56   |
|   math    |    0.0227     |  44   |
+-----------+---------------+-------+
2024-12-30 20:03:54,583 - evalscope - INFO - tag_level Report:
+----------------+---------------+-------+
|      tags      | average_score | count |
+----------------+---------------+-------+
|       en       |     0.26      |  100  |
| math&reasoning |     0.26      |  100  |
|   reasoning    |    0.4464     |  56   |
|      math      |    0.0227     |  44   |
+----------------+---------------+-------+
```