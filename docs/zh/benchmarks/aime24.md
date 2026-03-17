# AIME-2024


## 概述

AIME 2024（2024年美国邀请赛数学考试）是一个基于著名AIME竞赛题目的基准测试。这些题目代表了最具挑战性的高中数学问题，需要创造性的问题解决能力和高级数学推理能力。

## 任务描述

- **任务类型**：竞赛数学问题求解  
- **输入**：AIME级别的数学问题  
- **输出**：整数答案（0-999），附带逐步推理过程  
- **难度**：高级高中 / 初级大学水平  

## 主要特点

- 题目来自2024年AIME I和AIME II竞赛  
- 答案始终为0到999之间的整数  
- 要求具备创造性的数学推理与问题解决能力  
- 涉及主题：代数、几何、数论、组合数学、概率  
- 代表顶尖高中数学竞赛的难度水平  

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 答案应使用 `\boxed{}` 格式以便正确提取  
- 仅接受整数答案（符合AIME格式）  
- 题目难度显著高于GSM8K或标准MATH基准测试  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `aime24` |
| **数据集ID** | [evalscope/aime24](https://modelscope.cn/datasets/evalscope/aime24/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 30 |
| 提示词长度（平均） | 462.33 字符 |
| 提示词长度（最小/最大） | 242 / 1066 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "32187d6f",
      "content": "\nSolve the following math problem step by step. Put your answer inside \\boxed{}.\n\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the w ... [TRUNCATED 164 chars] ... g $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n\nRemember to put your answer inside \\boxed{}."
    }
  ],
  "target": "\\boxed{204}",
  "id": 0,
  "group_id": 0
}
```

## 提示模板

**提示模板：**
```text

Solve the following math problem step by step. Put your answer inside \boxed{{}}.

{question}

Remember to put your answer inside \boxed{{}}.
```

## 使用方法

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets aime24 \
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
    datasets=['aime24'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```