# General Decision

## 概述

评估包含每个记录一个选择、Noul 或评分问题的本地 JSONL 数据集。

## 任务描述

- **任务类型**：文本决策：选择、Noul 和评分
- **输入**：状态和一个带有命名选项或有序级别的问题
- **输出**：选择、是的概率或序数评分分布
- **领域**：文本理解和决策

## 主要特点

- 从通过 `dataset_args.general_decision.local_path` 传递的目录中读取 `default_test.jsonl`
- 字段：`id`，`state`，`question`（类型/说明/标准），`expected`；每个记录一个问题
- 保持选项顺序、源 ID 和评分描述
- 在其声明的能力范围内支持 System One 和原生 AutoTrust API

## 评估说明

- 基于规则的零样本评分；参考答案和来源仅用于评分，永远不会作为请求字段
- 默认的主要指标：离散参考标签上的准确性；连续评分运行可以选择 MAE
- 按类型分离概率指标；评分保留其期望值和归一化 RPS
- 无效响应被排除在普通指标之外，并反映在执行覆盖率中
- 这是一个本地数据入口点；无需上传
- 请参阅 [自定义决策数据集](../advanced_guides/custom_dataset/decision.md) 以获取可运行的 JSONL 示例、必需和可选字段、问题类型、指标、连续评分和配置
- 请参阅 [决策评估最佳实践](../best_practice/decision_evaluation.md) 以了解部署和 Web UI

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_decision` |
| **数据集ID** | `general_decision` |
| **论文** | N/A |
| **标签** | `Custom`, `Decision` |
| **指标** | `accuracy` |
| **默认射击次数** | 0-shot |
| **评估拆分** | `test` |

## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --eval-type systemone_api \
    --api-url http://127.0.0.1:18131/v1 \
    --api-key EMPTY_TOKEN \
    --datasets general_decision \
    --dataset-args '{"general_decision": {"local_path": "/path/to/data"}}' \
    --limit 10  # Remove this line for formal evaluation
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    api_key='EMPTY_TOKEN',
    datasets=['general_decision'],
    dataset_args={
        'general_decision': {
            'local_path': '/path/to/data',  # contains default_test.jsonl
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
