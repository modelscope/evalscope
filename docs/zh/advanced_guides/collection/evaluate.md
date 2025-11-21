# 用你的指数统一评测

一句话：使用混合 JSONL（由采样阶段生成）作为单一“数据集”运行评测，一次得到多层能力分数与加权指数总分。

## 评测配置

::::{tab-set}
:::{tab-item} 使用 `TaskConfig` 对象
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
:::{tab-item} 使用 CLI 命令行
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

## 注意事项
- datasets 固定使用 data_collection 代表“混合数据集入口”。
- dataset_args 中使用 local_path 指向你的 JSONL；可加 shuffle 控制顺序，默认不启用。
- 缓存、日志、报告目录结构：outputs/<timestamp>/{logs,predictions,reviews,reports,configs}。
- 批并发调优：eval_batch_size 控制推理批大小；评审并发由 judge_worker_num 控制。

## 指数加权回顾
最终指数得分：
$$
\mathbf{S} = \sum_{i=1}^{n} \boldsymbol{\alpha_i}\,\mathbf{s_i}, \quad \boldsymbol{\alpha_i} = \frac{w_i}{\sum_j w_j}
$$
其中 $s_i$ 为单数据集得分，$w_i$ 为展开后叶子权重（采样策略不会改变定义的权重）。

## 评测结果与解读
系统打印五层报告：
- **subset_level**：每个子集平均得分与数量
- **dataset_level**：每个数据集平均得分与数量
- **task_level**：每个任务平均得分与数量
- **tag_level**：每个标签平均得分与数量（Schema 名称追加到 tags）
- **category_level**：按自定义类别汇总的平均得分与数量（如有配置）

示例输出
```text
2025-11-21 17:27:32 - evalscope - INFO: subset_level Report:
+-----------+--------+--------------+---------------+---------------+-------+
| task_type | metric | dataset_name |  subset_name  | average_score | count |
+-----------+--------+--------------+---------------+---------------+-------+
| reasoning |  acc   |     arc      |   ARC-Easy    |     0.375     |   8   |
| reasoning |  acc   |     arc      | ARC-Challenge |      0.0      |   1   |
| reasoning |  acc   |    ceval     |     logic     |      1.0      |   1   |
+-----------+--------+--------------+---------------+---------------+-------+
2025-11-21 17:27:32 - evalscope - INFO: dataset_level Report:
+-----------+--------+--------------+---------------+-------+
| task_type | metric | dataset_name | average_score | count |
+-----------+--------+--------------+---------------+-------+
| reasoning |  acc   |     arc      |    0.3333     |   9   |
| reasoning |  acc   |    ceval     |      1.0      |   1   |
+-----------+--------+--------------+---------------+-------+
2025-11-21 17:27:32 - evalscope - INFO: task_level Report:
+-----------+--------+---------------+-------+
| task_type | metric | average_score | count |
+-----------+--------+---------------+-------+
| reasoning |  acc   |    0.7333     |  10   |
+-----------+--------+---------------+-------+
2025-11-21 17:27:32 - evalscope - INFO: tag_level Report:
+------+--------+---------------+-------+
| tags | metric | average_score | count |
+------+--------+---------------+-------+
|  en  |  acc   |    0.3333     |   9   |
|  zh  |  acc   |      1.0      |   1   |
+------+--------+---------------+-------+
2025-11-21 17:27:32 - evalscope - INFO: category_level Report:
+-----------------+--------+---------------+-------+
|    category0    | metric | average_score | count |
+-----------------+--------+---------------+-------+
| reasoning_index |  acc   |    0.7333     |  10   |
+-----------------+--------+---------------+-------+
```
## 常见问题
- 指数得分不符合预期：展开 Schema 验证归一化权重，确认未使用均匀采样误解权重效果。
- 修改只需重新运行：若不改 JSONL 内容可开启 --use-cache 节省推理时间。
- 大数据量评测速度慢：调大 eval_batch_size；确保模型/接口支持并发请求。

## 后续操作
- 可视化：evalscope app 打开交互界面筛选维度与模型。
- 版本化：为不同权重配置生成多份 JSONL，命名为 weighted_v1.jsonl / weighted_v2.jsonl 并分开评测。
