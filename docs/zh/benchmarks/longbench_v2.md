# LongBench-v2


## 概述

LongBench v2 是一个具有挑战性的基准测试，用于评估大语言模型对长上下文的理解能力。它涵盖了多种需要阅读和理解长文档（长度从数千到超过200万token不等）的真实世界任务，涉及多个领域，包括单文档问答、多文档问答、长上下文学习、长结构化数据理解以及代码仓库理解。

## 任务描述

- **任务类型**：长上下文多项选择题问答
- **输入**：长文档上下文 + 包含四个选项（A、B、C、D）的多项选择题
- **输出**：单个正确答案字母
- **领域**：单文档问答、多文档问答、长上下文学习、长结构化数据理解、代码仓库理解
- **难度**：简单 / 困难
- **长度**：短 / 中 / 长

## 主要特点

- 包含503道高质量问题，要求真正理解长文档内容
- 上下文长度从数千token到超过200万token不等
- 问题为双语（英文和中文）
- 设计上要求仔细阅读文档；若不阅读文档则无法猜出正确答案
- 覆盖多样化的现实应用场景

## 评估说明

- 默认配置使用 **0-shot** 评估（训练集用作测试集）
- 主要指标：**准确率**（答案字母完全匹配）
- 必须提供全部四个选项；无需随机打乱顺序
- 样本按上下文长度划分为 **3个子集**：`short`、`medium`、`long`
- 使用 `subset_list` 可评估特定长度子集（例如 `['short', 'medium']`）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `longbench_v2` |
| **数据集ID** | [ZhipuAI/LongBench-v2](https://modelscope.cn/datasets/ZhipuAI/LongBench-v2/summary) |
| **论文** | N/A |
| **标签** | `LongContext`, `MCQ`, `ReadingComprehension` |
| **指标** | `acc` |
| **默认Shots数** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 503 |
| 提示词长度（平均） | 872928.83 字符 |
| 提示词长度（最小/最大） | 49433 / 16184015 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `short` | 180 | 124200.42 | 49433 | 841252 |
| `medium` | 215 | 501002.72 | 172108 | 2233351 |
| `long` | 108 | 2861217.94 | 720823 | 16184015 |

## 样例示例

**子集**：`short`

```json
{
  "input": [
    {
      "id": "7e9a926f",
      "content": "Please read the following text and answer the questions below.\n\n<text>\nContents\nPreface.\n................................................................................................ 67\nI. China’s Court System and Reform Process.\n......... ... [TRUNCATED 163697 chars] ...  accelerate the construction of intelligent courts.\nC) Improve the work ability of office staff and strengthen the reserve of work knowledge.\nD) Use advanced information systems to improve the level of information technology in case handling."
    }
  ],
  "choices": [
    "Through technology empowerment, change the way of working and improve office efficiency.",
    "Establish new types of courts, such as intellectual property courts, financial courts, and Internet courts, and accelerate the construction of intelligent courts.",
    "Improve the work ability of office staff and strengthen the reserve of work knowledge.",
    "Use advanced information systems to improve the level of information technology in case handling."
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "short",
  "metadata": {
    "domain": "Single-Document QA",
    "sub_domain": "Financial",
    "difficulty": "easy",
    "length": "short",
    "context": "Contents\nPreface.\n................................................................................................ 67\nI. China’s Court System and Reform Process.\n.................................... 68\nII. Fully Implementing the Judicial Acco ... [TRUNCATED 162872 chars] ... a better environment for socialist rule of \nlaw, advance the judicial civilization to a higher level, and strive to make the \npeople obtain fair and just outcomes in every judicial case.\n法院的司法改革（2013-2018）.indd   161\n2019/03/01,星期五   17:42:05",
    "_id": "66f36490821e116aacb2cc22"
  }
}
```

## 提示模板

**提示模板：**
```text
Please read the following text and answer the questions below.

<text>
{document}
</text>

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 使用方法

### 使用命令行接口（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets longbench_v2 \
    --limit 10  # 正式评估时请删除此行
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['longbench_v2'],
    dataset_args={
        'longbench_v2': {
            # subset_list: ['short', 'medium', 'long']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```