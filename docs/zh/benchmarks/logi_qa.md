# LogiQA

## 概述

LogiQA 是一个用于评估逻辑推理能力的基准测试，其题目来源于专家编写的标准化考试试题，最初用于测试人类的逻辑推理能力。

## 任务描述

- **任务类型**：逻辑推理（多项选择题）
- **输入**：上下文段落、问题和 4 个选项
- **输出**：正确答案的字母（A、B、C 或 D）
- **重点**：演绎推理、逻辑推断

## 主要特点

- 由专家编写的逻辑推理题目
- 需要形式化的逻辑分析
- 考察多种推理模式（如三段论、条件推理等）
- 题目源自标准化考试
- 对人类和模型均具有挑战性

## 评估说明

- 默认配置使用 **0-shot** 评估
- 答案应遵循 "ANSWER: [LETTER]" 格式
- 在测试集（test split）上进行评估
- 使用简单准确率（accuracy）作为评估指标

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `logi_qa` |
| **数据集 ID** | [extraordinarylab/logiqa](https://modelscope.cn/datasets/extraordinarylab/logiqa/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估集** | `test` |
| **训练集** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 651 |
| 提示词长度（平均） | 1052.17 字符 |
| 提示词长度（最小/最大） | 389 / 1911 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "a7ac2424",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nIn the planning of a new district in a township, it w ... [TRUNCATED] ... ) Civic Park is north of the administrative service area.\nB) The leisure area is southwest of the cultural area.\nC) The cultural district is in the northeast of the business district.\nD) The business district is southeast of the leisure area."
    }
  ],
  "choices": [
    "Civic Park is north of the administrative service area.",
    "The leisure area is southwest of the cultural area.",
    "The cultural district is in the northeast of the business district.",
    "The business district is southeast of the leisure area."
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {}
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

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets logi_qa \
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
    datasets=['logi_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```