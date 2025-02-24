# 统一评测

在得到采样数据后，可以进行统一评测。

## 评测配置

配置评测任务，例如：

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

需要注意的是，其中：
- `datasets` 中指定的数据集名称固定为 `data_collection`，表示评测混合数据集
- `dataset_args` 中需要指定 `dataset_id`，表示评测数据集的路径，可以是本地路径，也可以是modelscope上的数据集id

## 评测结果

评测结果默认保存在 `outputs/` 目录下，包含4个层级的报告：

- `subset_level`：每个子集的平均得分和数量
- `dataset_level`：每个数据集的平均得分和数量
- `task_level`：每个任务的平均得分和数量
- `tag_level`：每个标签的平均得分和数量，schema的名称也作为标签，放在`tags`列中

例如，评测结果如下：

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