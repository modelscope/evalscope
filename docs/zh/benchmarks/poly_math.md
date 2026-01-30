# PolyMath


## 概述

PolyMath 是一个多语言数学推理基准测试，涵盖 18 种语言和 4 个难度级别，共包含 9,000 个高质量问题样本。该基准确保了难度的全面性、语言的多样性以及翻译的高质量，适用于具有区分度的多语言评估。

## 任务描述

- **任务类型**：多语言数学推理
- **输入**：18 种语言之一的数学问题
- **输出**：以 `\boxed{}` 格式表示的数值答案
- **领域**：涵盖多个难度级别和语言的数学问题

## 主要特性

- 支持 18 种语言：en、zh、ar、bn、de、es、fr、id、it、ja、ko、ms、pt、ru、sw、te、th、vi
- 4 个难度级别：low（低）、medium（中）、high（高）、top（顶级）
- 共计 9,000 个高质量问题
- 每个问题均配有语言特定的指令
- 高质量人工翻译，确保准确性

## 评估说明

- 默认评估使用 **test** 数据划分
- 主要指标：**Accuracy**（准确率），采用数值比较方式
- 附加指标：**DW-ACC**（难度加权准确率）
  - 权重分配：low=1，medium=2，high=4，top=8
  - 在不同难度级别间提供平衡的评分
- 结果按语言分别报告，并提供总体结果


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `poly_math` |
| **数据集 ID** | [evalscope/PolyMath](https://modelscope.cn/datasets/evalscope/PolyMath/summary) |
| **论文** | N/A |
| **标签** | `Math`, `MultiLingual`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 9,000 |
| 提示词长度（平均） | 342.15 字符 |
| 提示词长度（最小/最大） | 52 / 1536 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `en-low` | 125 | 292 | 142 | 600 |
| `zh-low` | 125 | 111.96 | 63 | 206 |
| `ar-low` | 125 | 259.15 | 138 | 536 |
| `bn-low` | 125 | 304.42 | 160 | 650 |
| `de-low` | 125 | 333.42 | 165 | 698 |
| `es-low` | 125 | 315.7 | 159 | 643 |
| `fr-low` | 125 | 331.06 | 178 | 634 |
| `id-low` | 125 | 332.51 | 175 | 691 |
| `it-low` | 125 | 315.19 | 164 | 661 |
| `ja-low` | 125 | 145.06 | 82 | 268 |
| `ko-low` | 125 | 163.17 | 89 | 342 |
| `ms-low` | 125 | 330.82 | 165 | 603 |
| `pt-low` | 125 | 306.37 | 160 | 655 |
| `ru-low` | 125 | 312.67 | 161 | 628 |
| `sw-low` | 125 | 324.54 | 169 | 638 |
| `te-low` | 125 | 311.38 | 161 | 575 |
| `th-low` | 125 | 256.28 | 124 | 519 |
| `vi-low` | 125 | 302.78 | 159 | 583 |
| `en-medium` | 125 | 304.88 | 107 | 823 |
| `zh-medium` | 125 | 182.79 | 52 | 503 |
| `ar-medium` | 125 | 282.52 | 98 | 794 |
| `bn-medium` | 125 | 323.46 | 110 | 761 |
| `de-medium` | 125 | 338.46 | 113 | 941 |
| `es-medium` | 125 | 322.59 | 120 | 785 |
| `fr-medium` | 125 | 330.45 | 116 | 766 |
| `id-medium` | 125 | 328.14 | 114 | 852 |
| `it-medium` | 125 | 315.01 | 110 | 772 |
| `ja-medium` | 125 | 210.79 | 68 | 548 |
| `ko-medium` | 125 | 219.33 | 64 | 547 |
| `ms-medium` | 125 | 314.84 | 95 | 829 |
| `pt-medium` | 125 | 314 | 111 | 767 |
| `ru-medium` | 125 | 334.75 | 120 | 828 |
| `sw-medium` | 125 | 335 | 110 | 899 |
| `te-medium` | 125 | 316.54 | 102 | 867 |
| `th-medium` | 125 | 276.01 | 84 | 658 |
| `vi-medium` | 125 | 307.78 | 108 | 820 |
| `en-high` | 125 | 391.3 | 120 | 1434 |
| `zh-high` | 125 | 212.87 | 70 | 1155 |
| `ar-high` | 125 | 356.49 | 115 | 1313 |
| `bn-high` | 125 | 414.23 | 132 | 1464 |
| `de-high` | 125 | 440.82 | 138 | 1483 |
| `es-high` | 125 | 422.2 | 134 | 1469 |
| `fr-high` | 125 | 428.81 | 133 | 1488 |
| `id-high` | 125 | 437.18 | 128 | 1536 |
| `it-high` | 125 | 408.41 | 128 | 1445 |
| `ja-high` | 125 | 246.59 | 84 | 1206 |
| `ko-high` | 125 | 261.16 | 98 | 1195 |
| `ms-high` | 125 | 412.78 | 55 | 1454 |
| `pt-high` | 125 | 408.39 | 127 | 1414 |
| `ru-high` | 125 | 426.44 | 144 | 1476 |
| `sw-high` | 125 | 438.1 | 125 | 1476 |
| `te-high` | 125 | 405.18 | 126 | 1430 |
| `th-high` | 125 | 351.18 | 108 | 1345 |
| `vi-high` | 125 | 383.09 | 124 | 1442 |
| `en-top` | 125 | 420.59 | 141 | 1346 |
| `zh-top` | 125 | 220.16 | 73 | 876 |
| `ar-top` | 125 | 378.14 | 136 | 1238 |
| `bn-top` | 125 | 443.98 | 160 | 1392 |
| `de-top` | 125 | 470.34 | 169 | 1432 |
| `es-top` | 125 | 456.15 | 150 | 1432 |
| `fr-top` | 125 | 464.7 | 153 | 1457 |
| `id-top` | 125 | 469.23 | 151 | 1478 |
| `it-top` | 125 | 445.74 | 146 | 1400 |
| `ja-top` | 125 | 259.17 | 85 | 925 |
| `ko-top` | 125 | 277.8 | 89 | 968 |
| `ms-top` | 125 | 458.26 | 144 | 1521 |
| `pt-top` | 125 | 444.11 | 144 | 1407 |
| `ru-top` | 125 | 466.7 | 159 | 1440 |
| `sw-top` | 125 | 469.38 | 147 | 1452 |
| `te-top` | 125 | 431.14 | 147 | 1323 |
| `th-top` | 125 | 384.58 | 137 | 1154 |
| `vi-top` | 125 | 423.55 | 154 | 1352 |

## 样例示例

**子集**: `en-low`

```json
{
  "input": [
    {
      "id": "8ac6f5ab",
      "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nNote: Please put the final answer in the $\\boxed\\{\\}$."
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "level": "low",
    "language": "en",
    "index": "0"
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets poly_math \
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
    datasets=['poly_math'],
    dataset_args={
        'poly_math': {
            # subset_list: ['en-low', 'zh-low', 'ar-low']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```