# 用你的指数统一评测
## 评测配置（Python）
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen2.5',
    eval_type='openai_api',  # 推荐：OpenAI 兼容 API
    api_url='http://127.0.0.1:8801/v1',
    api_key='EMPTY',
    datasets=['data_collection'],  # 固定名称：混合数据集入口
    dataset_args={'data_collection': {'dataset_id': 'outputs/rag_index.jsonl'}},  # 你的采样 JSONL（本地路径或云端 ID）
)
run_task(task_cfg=task_cfg)
```

## 评测配置（CLI）
```bash
evalscope eval \
  --model qwen2.5 \
  --eval-type openai_api \
  --api-url http://127.0.0.1:8801/v1 \
  --api-key EMPTY \
  --datasets data_collection \
  --dataset-args '{"data_collection": {"dataset_id":"outputs/rag_index.jsonl"}}'
```

## 注意事项
- `datasets` 必须为 `data_collection`，表示混合数据集入口。
- `dataset_id` 指向你的采样 JSONL（本地或平台 ID）。
- 输出目录结构由 `OutputsStructure` 管理。

## 评测结果与解读
系统打印四层报告，并生成 JSON：
- **subset_level**：每个子集平均得分与数量
- **dataset_level**：每个数据集平均得分与数量
- **task_level**：每个任务平均得分与数量
- **tag_level**：每个标签平均得分与数量（Schema 名称追加到 tags）

## 示例输出
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

## 从报告到最终 Index 分数
- 读取 **dataset_level**：用归一化权重 α_i 线性加权 Σ(α_i · s_i) 得到你的 **Index**。
- 聚焦能力：基于 **tag_level** 过滤如 `rag` 标签再加权。

## 提示
- 并发：预测 `eval_batch_size`，评审 `judge_worker_num`。
- 复现：`use_cache` 复用生成；`rerun_review` 仅重算评分。

