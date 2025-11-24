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

## 使用约定
- datasets 固定使用 data_collection 作为混合入口。
- dataset_args.local_path 指向混合 JSONL；可选 shuffle 控制样本遍历顺序。
- 输出结构：outputs/<timestamp>/{logs,predictions,reviews,reports,configs}。
- 推理并发：eval_batch_size；评审并发：judge_worker_num。

## 指标与输出

系统会在五个“观察视角”输出报告：
1. subset_level：原始数据集中子集（如 ARC-Challenge / ARC-Easy）。
2. dataset_level：同一数据集所有子集汇总（如 arc）。
3. task_level：同一任务类型（如 reasoning）。
4. tag_level：样本 tags 维度（语言、模式等；Schema 名称可并入 tags）。
5. category_level：自定义分组（如 reasoning_index）。

指标说明：
- micro_avg.：逐样本加总后计算整体准确率（总正确数 / 总样本数）。样本数高的子集影响更大。
- macro_avg.：对该层“成员”（子集）各自得分简单平均，每个成员同权，不受样本数影响。
- weighted_avg.：按定义的叶子权重 $w_i$ 做加权（权重在采样前定义，采样过程不改变权重含义）。


示例输出：
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
## 常见问题
- 指数得分不符合预期：展开 Schema 验证归一化权重，确认未使用均匀采样误解权重效果。
- 修改只需重新运行：若不改 JSONL 内容可开启 --use-cache 节省推理时间。
- 大数据量评测速度慢：调大 eval_batch_size；确保模型/接口支持并发请求。

## 后续操作
- 可视化：evalscope app 打开交互界面筛选维度与模型。
- 版本化：为不同权重配置生成多份 JSONL，命名为 weighted_v1.jsonl / weighted_v2.jsonl 并分开评测。
