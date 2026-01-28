# GSM8K

## 概述

GSM8K（Grade School Math 8K）是一个高质量的数据集，包含由人工编写者创作的 8.5K 道语言多样化的中小学数学应用题。该数据集专门用于评估和提升语言模型的多步数学推理能力。

## 任务描述

- **任务类型**：数学应用题求解  
- **输入**：自然语言描述的数学应用题  
- **输出**：通过逐步推理得出的数值答案  
- **难度**：小学水平（需 2–8 步推理）

## 主要特点

- 问题涉及基本算术运算（加、减、乘、除）  
- 解答过程包含 2 到 8 个连续推理步骤  
- 问题表述具有高度的语言多样性  
- 人工编写确保自然语言质量  
- 答案为明确的数值，便于客观评估  

## 评估说明

- 默认配置使用 **4-shot** 示例并结合思维链（Chain-of-Thought, CoT）提示  
- 答案应使用 `\boxed{}` 格式包裹以便正确提取  
- 评估指标会提取数值进行准确率比较  
- 支持零样本（zero-shot）和少样本（few-shot）两种评估模式  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gsm8k` |
| **数据集ID** | [AI-ModelScope/gsm8k](https://modelscope.cn/datasets/AI-ModelScope/gsm8k/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2110.14168) |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认样本数** | 4-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,319 |
| 提示词长度（平均） | 1966.87 字符 |
| 提示词长度（最小/最大） | 1800 / 2575 字符 |

## 样例示例

**子集**: `main`

```json
{
  "input": [
    {
      "id": "0bc7f97b",
      "content": "Here are some examples of how to solve similar problems:\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\nReasoning:\nNatalia sold 48/ ... [TRUNCATED] ... ds every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market."
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

<details>
<summary>少样本模板</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}

{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

</details>

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gsm8k \
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
    datasets=['gsm8k'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```