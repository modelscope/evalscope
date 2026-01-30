# MathQA

## 概述

MathQA 是一个大规模数学应用题求解数据集，通过对 AQuA-RAT 数据集进行标注而构建，使用了一种新的表示语言为每个问题提供了完整的可执行操作程序。该数据集包含多样化的数学问题，需要多步推理才能解答。

## 任务描述

- **任务类型**：数学推理（多项选择题）
- **输入**：带有多个选项的数学应用题
- **输出**：正确答案及逐步推理过程（Chain-of-Thought）
- **难度**：多样（从小学到中级水平）

## 主要特点

- 标注了可执行的操作程序
- 考察量化推理与问题解决能力
- 涵盖多种数学主题和题型
- 多项选择格式，附带结构化解法
- 适用于评估模型的数学推理能力

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 使用思维链（Chain-of-Thought, CoT）提示进行推理
- 在测试集（test split）上进行评估
- 使用简单准确率（accuracy）作为评估指标
- 推理步骤可在元数据（metadata）中获取

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `math_qa` |
| **数据集ID** | [extraordinarylab/math-qa](https://modelscope.cn/datasets/extraordinarylab/math-qa/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,985 |
| 提示词长度（平均） | 433.02 字符 |
| 提示词长度（最小/最大） | 257 / 879 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "e77fca06",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E. Think step by step before answering.\n\na shopkeeper sold an article offering a discount of 5 % and earned a profit of 31.1 % . what would have been the percentage of profit earned if no discount had been offered ?\n\nA) 38\nB) 27.675\nC) 30\nD) data inadequate\nE) none of these"
    }
  ],
  "choices": [
    "38",
    "27.675",
    "30",
    "data inadequate",
    "none of these"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": "\"giving no discount to customer implies selling the product on printed price . suppose the cost price of the article is 100 . then printed price = 100 ã — ( 100 + 31.1 ) / ( 100 â ˆ ’ 5 ) = 138 hence , required % profit = 138 â € “ 100 = 38 % answer a\""
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets math_qa \
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
    datasets=['math_qa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```