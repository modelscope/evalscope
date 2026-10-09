# BANKING77


## 概述

使用完整的 77 类分类体系进行细粒度的银行客服意图分类。

## 任务描述

- **任务类型**：单选分类
- **输入**：银行客服消息
- **输出**：77 种银行意图之一
- **领域**：客户服务

## 主要特性

- 公开数据集：ModelScope 上的 `mteb/banking77`
- 保留原始标签和完整的任务上下文
- 支持聊天生成模型和文本单选（System One Choice）模型

## 评估说明

- 报告指标为准确率（accuracy），而非 macro-F1。每个样本均可从全部 77 个意图中选择；不进行候选剪枝，也不额外添加“超出范围”（out-of-scope）意图。
- 默认采用 0-shot 设置；若存在训练集，可配置训练示例
- 系统提示将转换为单选任务指令，不使用原生的聊天角色层级结构
- 评估语义版本：v1.0

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `banking77` |
| **数据集ID** | [mteb/banking77](https://modelscope.cn/datasets/mteb/banking77/summary) |
| **论文** | N/A |
| **标签** | `MCQ` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估集** | `test` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,076 |
| 提示词长度（平均） | 2354.28 字符 |
| 提示词长度（最小/最大） | 2313 / 2668 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "59e429c3",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,1,2,3,4,5,6 ... [TRUNCATED 1840 chars] ... e to verify identity\n44) verify my identity\n45) verify source of funds\n46) verify top up\n47) virtual card not working\n48) visa or mastercard\n49) why verify identity\n50) wrong amount of cash received\n51) wrong exchange rate for cash withdrawal"
    }
  ],
  "choices": [
    "activate my card",
    "age limit",
    "apple pay or google pay",
    "atm support",
    "automatic top up",
    "balance not updated after bank transfer",
    "balance not updated after cheque or cash deposit",
    "beneficiary not allowed",
    "cancel transfer",
    "card about to expire",
    "... [TRUNCATED 67 more items] ..."
  ],
  "target": "L",
  "id": 0,
  "group_id": 0
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets banking77 \
    --limit 10  # 正式评估时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['banking77'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
