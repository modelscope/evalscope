# General Decision


## 概述

用户提供的本地 JSONL 文件中的单个问题类型决策。

## 任务描述

- **任务类型**：类型文本决策
- **输入**：状态和一个带有命名选项或有序级别的问题
- **输出**：选择、肯定的概率或序数得分分布
- **领域**：文本理解和决策

## 主要特点

- 从通过 `dataset_args.general_decision.local_path` 传递的目录中读取 `default_test.jsonl`
- 字段：`id`，`state`，`question`（类型/说明/标准），`expected`；每条记录一个问题
- 保持选项顺序、源 ID 和评分描述
- 支持 System One 和在其声明的能力范围内的原生 AutoTrust API

## 评估说明

- 零样本确定性评分；黄金答案和来源永远不会进入请求
- 主要指标：准确性；Choice/Noul 使用分类黄金标准，Score 使用整数级别
- 概率指标按类型分开；Score 保留其期望值和归一化的 RPS
- 无效响应被排除在普通指标之外，并反映在执行覆盖率中
- 这是一个本地数据入口点；无需上传
- 请参阅决策评估最佳实践以获取示例和部署限制


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `general_decision` |
| **数据集ID** | `general_decision` |
| **论文** | N/A |
| **标签** | `Custom`, `Decision` |
| **指标** | `accuracy` |
| **默认样本数** | 0-shot |
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
