# BoolQ


## 概述

使用段落和自然出现的问题进行二元阅读理解。

## 任务描述

- **任务类型**: 单选分类
- **输入**: 段落和是/否问题
- **输出**: 是或否
- **领域**: 英语阅读理解

## 主要特点

- 公开数据集: ModelScope 上的 `google/boolq`
- 保留源标签和完整的任务上下文
- 支持聊天生成和文本系统单选模型

## 评估说明

- 使用公开的验证标签，默认为零样本。决策 API 默认使用双选项选择。设置 `extra_params.decision_kind="noul"` 以获取 P(yes)，默认 `decision_threshold=0.5`。
- 默认为 0-shot；在有训练分割的情况下，可以配置训练示例
- 系统提示变为选择任务指令，没有原生的聊天角色层次结构
- 评估语义版本: v1.1


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `boolq` |
| **数据集ID** | [google/boolq](https://modelscope.cn/datasets/google/boolq/summary) |
| **论文** | N/A |
| **标签** | `MCQ` |
| **指标** | `accuracy` |
| **默认样本数** | 0-shot |
| **评估分割** | `validation` |
| **训练分割** | `train` |


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
      "id": "47f2b621",
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

*注：部分内容因显示需要被截断。*

## 提示模板

**提示模板:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

{question}

{choices}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `decision_kind` | `str` | `choice` | 决策 API 问题类型：选择（默认）或 noul。noul 需要一个决策 API。 |
| `decision_threshold` | `float` | `0.5` | 当 P(yes) 至少达到此阈值时，将 Noul 分类为 yes，范围在 [0, 1] 之间。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets boolq \
    --limit 10  # Remove this line for formal evaluation
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
    dataset_args={
        'boolq': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
