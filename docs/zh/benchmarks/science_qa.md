# ScienceQA


## 概述

ScienceQA 是一个多模态基准测试，包含来自小学和高中课程的多项选择科学问题。它涵盖了自然科学、社会科学和语言科学等多个学科，并提供可选的文本和图像上下文。

## 任务描述

- **任务类型**: 多模态科学问答
- **输入**: 问题和多个选项，以及可选的文本 (`hint`) 和图像上下文
- **输出**: 正确答案选项字母
- **领域**: 自然科学、社会科学、语言科学

## 主要特点

- 问题来源于真实的 K-12 科学课程
- 文本和图像上下文是可选的，并且因问题而异
- 带有详细的背景知识和解题解释注释
- 支持对思维链推理的研究
- 覆盖多个年级和难度范围
- 丰富的元数据，包括主题、技能和类别信息

## 评估说明

- 默认评估使用 **test** 分割
- 主要指标: 多项选择题的 **准确率**
- 使用 Chain-of-Thought (CoT) 提示进行推理
- 评估版本 **v1.1** 在问题前添加非空 `hint` 文本作为 `Context: ...`；空提示不添加上下文
- `lecture` 和 `solution` 注释作为元数据保留，并从评估提示中排除
- 问题涵盖从小学到高中的各个年级


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `science_qa` |
| **数据集ID** | [AI-ModelScope/ScienceQA](https://modelscope.cn/datasets/AI-ModelScope/ScienceQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,241 |
| 提示词长度（平均） | 472.92 字符 |
| 提示词长度（最小/最大） | 255 / 2457 字符 |

**图像统计:**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 2,017 |
| 每个样本的图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 170x77 - 750x625 |
| 格式 | png |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "266efb09",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B. Think step by step before answering.\n\nWhich figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n—Homer, The Iliad\n\nA) chiasmus\nB) apostrophe"
        }
      ]
    }
  ],
  "choices": [
    "chiasmus",
    "apostrophe"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "hint": "",
    "task": "closed choice",
    "grade": "grade11",
    "subject": "language science",
    "topic": "figurative-language",
    "category": "Literary devices",
    "skill": "Classify the figure of speech: anaphora, antithesis, apostrophe, assonance, chiasmus, understatement",
    "lecture": "Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\nAnaphora is the repetition of the same word or words at the beginning of several phrases or clauses.\nWe are united ... [TRUNCATED 446 chars] ... but reverses the order of words.\nNever let a fool kiss you or a kiss fool you.\nUnderstatement involves deliberately representing something as less serious or important than it really is.\nAs you know, it can get a little cold in the Antarctic.",
    "solution": "The text uses apostrophe, a direct address to an absent person or a nonhuman entity.\nO goddess is a direct address to a goddess, a nonhuman entity."
  }
}
```

*注意: 部分内容为了显示被截断。*

## 提示模板

**提示模板:**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets science_qa \
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
    datasets=['science_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
