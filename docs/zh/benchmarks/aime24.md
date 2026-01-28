# AIME-2024

## 概述

AIME 2024（2024年美国数学邀请赛）是一个基于著名AIME竞赛题目的基准测试。这些题目代表了最具挑战性的高中数学问题，要求具备创造性的问题解决能力和高级数学推理能力。

## 任务描述

- **任务类型**：竞赛数学问题求解  
- **输入**：AIME级别的数学问题  
- **输出**：整数答案（0-999），附带逐步推理过程  
- **难度**：高级高中 / 初级大学水平  

## 主要特点

- 题目来自2024年AIME I和AIME II竞赛  
- 答案始终为0到999之间的整数  
- 需要创造性数学推理与问题解决能力  
- 涉及主题：代数、几何、数论、组合数学、概率  
- 代表顶尖高中数学竞赛难度  

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 答案应使用 `\boxed{}` 格式以便正确提取  
- 仅接受整数答案（符合AIME格式）  
- 题目难度显著高于GSM8K或标准MATH基准  
- 元数据中提供参考解答用于分析  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `aime24` |
| **数据集ID** | [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 30 |
| 提示词长度（平均） | 405.33 字符 |
| 提示词长度（最小/最大） | 185 / 1009 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "2f896eed",
      "content": "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she wa ... [TRUNCATED] ... e coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "204",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_id": 60,
    "solution": "$\\frac{9}{s} + t = 4$ in hours and $\\frac{9}{s+2} + t = 2.4$ in hours.\nSubtracting the second equation from the first, we get, \n$\\frac{9}{s} - \\frac{9}{s+2} = 1.6$\nMultiplying by $(s)(s+2)$, we get \n$9s+18-9s=18=1.6s^{2} + 3.2s$\nMultiplying b ... [TRUNCATED] ... s = 2.5$. Now, $2.5+0.5 = 3$. Taking $\\frac{9}{3} = 3$, we find that it will take three hours for the 9 kilometers to be traveled. The t minutes spent at the coffeeshop can be written as $144-48(2.5)$, so t = 24. $180 + 24 = 204$. -sepehr2010"
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