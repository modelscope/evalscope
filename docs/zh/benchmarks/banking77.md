# BANKING77


## 概述

具有完整77类分类法的细粒度银行支持意图分类。

## 任务描述

- **任务类型**：单选分类
- **输入**：银行支持消息
- **输出**：77个银行意图之一
- **领域**：客户服务

## 主要特点

- 公共数据集：ModelScope上的`mteb/banking77`
- 保留源标签和完整的任务上下文
- 支持聊天生成和文本系统单选模型

## 评估说明

- 报告准确率；决策API还报告宏观F1和概率指标。所有77个意图对每个样本都可用；没有候选修剪或添加超出范围的意图。
- 默认为0-shot；在有训练分割的情况下可以配置训练示例
- 系统提示成为选择任务指令，没有原生的聊天角色层次结构
- 评估语义版本：v1.1


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `banking77` |
| **数据集ID** | [mteb/banking77](https://modelscope.cn/datasets/mteb/banking77/summary) |
| **论文** | N/A |
| **标签** | `MCQ` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |


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
      "id": "7b8ddfa9",
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

*注：部分内容为了显示被截断。*

## 提示模板

**提示模板:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets banking77 \
    --limit 10  # Remove this line for formal evaluation
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['banking77'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
