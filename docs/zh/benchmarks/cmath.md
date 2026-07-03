# CMATH

## 概述

CMATH 是一个面向中国小学数学的基准测试，包含 1,698 道覆盖 1 至 6 年级的数学题目。该基准用于评估语言模型在中文数学应用题上的数学推理能力，题目难度随年级递增。

## 任务描述

- **任务类型**：中文数学应用题求解  
- **输入**：中文数学应用题（小学水平）  
- **输出**：分步推理过程及最终数值答案  
- **难度**：1 年级（最简单）至 6 年级（最难）

## 主要特点

- 包含 1,098 道测试题 + 600 道验证题  
- 六个年级（1–6）划分，便于细粒度难度分析  
- 题目为中文，测试语言特定的推理能力  
- 答案形式为简单数值（整数或小数）  
- 元数据包含推理步骤数量和数字位数复杂度

## 评估说明

- 默认配置使用 **0-shot** 评估  
- 答案需用 `\boxed{}` 格式包裹以便正确提取  
- 使用数值准确率（numeric accuracy）作为答案比对指标  
- 结果可按年级细分分析  
- 默认使用中文提示模板

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cmath` |
| **数据集ID** | [evalscope/cmath](https://modelscope.cn/datasets/evalscope/cmath/summary) |
| **论文** | N/A |
| **标签** | `Chinese`, `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,098 |
| 提示词长度（平均） | 70.14 字符 |
| 提示词长度（最小/最大） | 38 / 191 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Grade 1` | 164 | 60.65 | 45 | 99 |
| `Grade 2` | 253 | 63.3 | 38 | 168 |
| `Grade 3` | 237 | 69.49 | 49 | 148 |
| `Grade 4` | 120 | 75.33 | 51 | 114 |
| `Grade 5` | 126 | 76.79 | 49 | 124 |
| `Grade 6` | 198 | 80.16 | 42 | 191 |

## 样例示例

**子集**: `Grade 1`

```json
{
  "input": [
    {
      "id": "cc99ea45",
      "content": "妈咪买了3盒茶叶，一盒茶叶有6小包，一共买了多少小包茶叶？\n请一步一步推理，最后将答案放在\\boxed{}中。\n"
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "subset_key": "Grade 1",
  "metadata": {
    "reasoning_step": 1,
    "num_digits": 2
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
请一步一步推理，最后将答案放在\boxed{{}}中。

```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cmath \
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
    datasets=['cmath'],
    dataset_args={
        'cmath': {
            # subset_list: ['Grade 1', 'Grade 2', 'Grade 3']  # 可选，用于评估指定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```