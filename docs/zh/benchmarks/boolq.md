# BoolQ


## 概述

基于段落和自然出现的问题进行二元阅读理解。

## 任务描述

- **任务类型**：单选分类
- **输入**：段落和是非问题
- **输出**：是 或 否
- **领域**：英文阅读理解

## 主要特性

- 公开数据集：ModelScope 上的 `google/boolq`
- 保留原始标签和完整的任务上下文
- 支持聊天生成和文本 System One Choice 模型

## 评估说明

- 默认使用公开的验证集标签，采用零样本（zero-shot）设置。System One 评估的是二选项选择题，而非某些 Jev 研究中使用的 Noul 协议。
- 默认为 0-shot；若存在训练集，可配置训练示例
- 系统提示被转换为选择题任务指令，不包含原生的聊天角色层级结构
- 评估语义版本：v1.0


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `boolq` |
| **数据集ID** | [google/boolq](https://modelscope.cn/datasets/google/boolq/summary) |
| **论文** | 无 |
| **标签** | `MCQ` |
| **指标** | `accuracy` |
| **默认样本数（Shots）** | 0-shot |
| **评估划分** | `validation` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,270 |
| 提示词长度（平均） | 822.26 字符 |
| 提示词长度（最小/最大） | 300 / 5043 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "800c0332",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nPassage:\nAll biomass goes through at least some of these  ... [TRUNCATED 1152 chars] ... versity of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.\n\nQuestion: does ethanol take more energy make that produces\n\nA) Yes\nB) No"
    }
  ],
  "choices": [
    "Yes",
    "No"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0
}
```

*注：部分内容为显示目的已截断。*

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
    --datasets boolq \
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
    datasets=['boolq'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
