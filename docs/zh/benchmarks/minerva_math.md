# Minerva-Math


## 概述

Minerva-Math 是一个用于评估语言模型高级数学与定量推理能力的基准测试。该基准包含 272 道具有挑战性的问题，主要来源于 MIT OpenCourseWare 课程，涵盖大学及研究生级别的 STEM 学科。

## 任务描述

- **任务类型**：高级 STEM 问题求解  
- **输入**：大学/研究生级别的数学或科学问题  
- **输出**：包含逐步推导过程和最终答案的解答  
- **难度**：大学至研究生级别  

## 主要特点

- 包含 272 道来自 MIT OpenCourseWare 的难题  
- 涵盖高级学科：固体化学、天文学、微分方程、狭义相对论等  
- 难度为大学及研究生级别  
- 测试深层次的数学与科学推理能力  
- 问题需要多步定量推理  

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 答案应使用 `\boxed{}` 格式包裹，以便正确提取  
- 使用 LLM-as-judge 对复杂答案进行评估  
- 问题可能需要特定领域的知识（如物理、化学等）  
- 旨在测试模型推理能力的上限  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `minerva_math` |
| **数据集ID** | [knoveleng/Minerva-Math](https://modelscope.cn/datasets/knoveleng/Minerva-Math/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 272 |
| 提示词长度（平均） | 492.27 字符 |
| 提示词长度（最小/最大） | 129 / 1069 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "715a0fa1",
      "content": "Each of the two Magellan telescopes has a diameter of $6.5 \\mathrm{~m}$. In one configuration the effective focal length is $72 \\mathrm{~m}$. Find the diameter of the image of a planet (in $\\mathrm{cm}$ ) at this focus if the angular diameter of the planet at the time of the observation is $45^{\\prime \\prime}$.\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "Start with:\n\\[\ns=\\alpha f \\text {, }\n\\]\nwhere $s$ is the diameter of the image, $f$ the focal length, and $\\alpha$ the angular diameter of the planet. For the values given in the problem:\n\\[\ns=\\frac{45}{3600} \\frac{\\pi}{180} 7200=\\boxed{1.6} \\mathrm{~cm}\n\\]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "type": "Introduction to Astronomy (8.282J Spring 2006)",
    "idx": 0
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets minerva_math \
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
    datasets=['minerva_math'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```