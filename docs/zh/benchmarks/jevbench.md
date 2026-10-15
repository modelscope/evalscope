# JevBench public subset


## 概述

公开的231个Benchmark Heaven决策，保留了原始的评分标准和标签。

## 任务描述

- **任务类型**: 文本决策
- **输入**: 状态和一个带有命名选项或有序级别的问题
- **输出**: 选择、是的概率或序数得分分布
- **领域**: 文本理解和决策

## 主要特点

- 固定的Benchmark Heaven公开快照：72个原始任务，48个简单任务和111个困难任务（139个选择题，74个Noul题，18个评分题）
- 来源: [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench/tree/609a664e81f46e03731600bd7bd9f1b7a61257c5); MIT许可的公开记录
- 保留选项顺序、来源ID和评分标准描述
- 支持System One和原生AutoTrust API在其声明的能力范围内

## 评估说明

- 零样本确定性评分；黄金答案和来源从未进入请求
- 主要指标：公共子集准确率（缺失的预测计为错误）；还包括家族宏观准确率、每种类型的概率指标和评分期望值误差
- 四级和五级评分任务保留其原始评分标准；AutoTrust固定六级服务明确拒绝这些任务
- 无效响应被排除在普通指标之外，并反映在执行覆盖率中
- 公开评分器：概率总和容差0.001；仅在0.02范围内进行归一化；Noul平局选择否
- 此公开快照不产生完整的JevBench复合或能力排行榜指标

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `jevbench` |
| **数据集ID** | [evalscope/jevbench](https://modelscope.cn/datasets/evalscope/jevbench/summary) |
| **论文** | N/A |
| **标签** | `Decision` |
| **指标** | `public_accuracy` |
| **默认样本数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 231 |
| 提示词长度（平均） | 2521.31 字符 |
| 提示词长度（最小/最大） | 224 / 16126 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "f98fca19",
      "content": "{\"state\": \"Where is my package? I ordered it last week and it still hasn't arrived.\", \"question\": {\"criteria\": {\"billing_question\": \"Asks about a charge, invoice or payment\", \"cancel_order\": \"Wants to cancel an order\", \"change_address\": \"Wants to change the delivery address\", \"report_damage\": \"Received an item that is broken or damaged\", \"track_order\": \"Wants to know where an order is or when it arrives\"}, \"instructions\": \"Which intent does the user's message express?\", \"type\": \"choice\"}}",
      "internal": {
        "systemone": {
          "kind": "choice",
          "state": "Where is my package? I ordered it last week and it still hasn't arrived.",
          "instructions": "Which intent does the user's message express?",
          "criteria": {
            "billing_question": "Asks about a charge, invoice or payment",
            "cancel_order": "Wants to cancel an order",
            "change_address": "Wants to change the delivery address",
            "report_damage": "Received an item that is broken or damaged",
            "track_order": "Wants to know where an order is or when it arrives"
          },
          "answer_prefix": "ANSWER: "
        }
      }
    }
  ],
  "target": "track_order",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "source_id": "easy-intent-00",
    "decision_gold": "track_order",
    "decision_kind": "choice",
    "source_group": null,
    "family": "intent",
    "tier": "",
    "language": "",
    "continuous": false,
    "decision_labels": [
      "billing_question",
      "cancel_order",
      "change_address",
      "report_damage",
      "track_order"
    ],
    "fixed_classes": false
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --eval-type systemone_api \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets jevbench \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['jevbench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
