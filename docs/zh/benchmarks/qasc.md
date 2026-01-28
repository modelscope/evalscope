# QASC

## 概述

QASC（Question Answering via Sentence Composition，基于句子组合的问题回答）是一个专注于多跳句子组合的问题回答数据集。它包含 9,980 个面向小学科学的 8 选 1 多项选择题，要求模型组合多个事实才能得出正确答案。

## 任务描述

- **任务类型**：多跳科学问答（多项选择）
- **输入**：包含 8 个选项的科学问题
- **输出**：正确答案对应的字母
- **重点**：句子组合与多跳推理

## 主要特点

- 包含 9,980 个小学科学问题
- 采用 8 选 1 的多项选择格式
- 需要组合两个事实才能作答
- 测试模型在科学知识上的多跳推理能力
- 每个问题均标注了支持性事实

## 评估说明

- 默认配置使用 **0-shot** 评估
- 在验证集（validation split）上进行评估
- 使用简单准确率（accuracy）作为评估指标
- 适用于评估组合推理能力

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `qasc` |
| **数据集ID** | [extraordinarylab/qasc](https://modelscope.cn/datasets/extraordinarylab/qasc/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 926 |
| 提示词长度（平均） | 362.58 字符 |
| 提示词长度（最小/最大） | 283 / 505 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "d92a545e",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H.\n\nClimate is generally described in terms of what?\n\nA) sand\nB) occurs over a wide range\nC) forests\nD) Global warming\nE) rapid changes occur\nF) local weather conditions\nG) measure of motion\nH) city life"
    }
  ],
  "choices": [
    "sand",
    "occurs over a wide range",
    "forests",
    "Global warming",
    "rapid changes occur",
    "local weather conditions",
    "measure of motion",
    "city life"
  ],
  "target": "F",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

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
    --datasets qasc \
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
    datasets=['qasc'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```