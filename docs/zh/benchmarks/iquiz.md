# IQuiz

## 概述

IQuiz 是一个中文基准测试，用于评估 AI 模型在智商（IQ）和情商（EQ）问题上的表现。它通过多项选择题来测试逻辑推理、模式识别以及社会情感理解能力。

## 任务描述

- **任务类型**：多项选择题问答
- **输入**：中文问题及多个选项
- **输出**：所选答案及解释（思维链，Chain-of-Thought）
- **语言**：中文

## 主要特点

- 同时评估 IQ 和 EQ 能力
- 中文认知能力测评
- 多种难度级别
- 要求在选择答案的同时提供解释
- 测试逻辑推理与情感理解能力

## 评估说明

- 默认配置使用 **0-shot** 评估
- 主要指标：**准确率（Accuracy）**
- 子集：**IQ**（逻辑推理）和 **EQ**（情感智能）
- 使用中文思维链（Chain-of-Thought）提示模板
- 在 **test** 数据划分上进行评估
- 元数据包含难度级别信息

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `iquiz` |
| **数据集 ID** | [AI-ModelScope/IQuiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary) |
| **论文** | N/A |
| **标签** | `Chinese`, `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估数据划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 120 |
| 提示词长度（平均） | 248.31 字符 |
| 提示词长度（最小/最大） | 146 / 394 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `IQ` | 40 | 194.5 | 146 | 323 |
| `EQ` | 80 | 275.21 | 219 | 394 |

## 样例示例

**子集**: `IQ`

```json
{
  "input": [
    {
      "id": "748f2700",
      "content": "回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：\"答案：[LETTER]\"（不带引号），其中 [LETTER] 是 A,B,C,D 中的一个。请在回答前进行一步步思考。\n\n问题：天气预报说本周星期三会下雨，昨天果然下雨了，今天星期几？\n选项：\nA) 星期一\nB) 星期二\nC) 星期三\nD) 星期四\n"
    }
  ],
  "choices": [
    "星期一",
    "星期二",
    "星期三",
    "星期四"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "level": 1
  }
}
```

## 提示模板

**提示模板：**
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets iquiz \
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
    datasets=['iquiz'],
    dataset_args={
        'iquiz': {
            # subset_list: ['IQ', 'EQ']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```