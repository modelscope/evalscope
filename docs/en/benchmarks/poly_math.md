# PolyMath


## Overview

PolyMath is a multilingual mathematical reasoning benchmark covering 18 languages and 4 difficulty levels with 9,000 high-quality problem samples. It ensures difficulty comprehensiveness, language diversity, and high-quality translation for discriminative multilingual evaluation.

## Task Description

- **Task Type**: Multilingual Mathematical Reasoning
- **Input**: Math problem in one of 18 languages
- **Output**: Numerical answer in \boxed{} format
- **Domains**: Mathematics across multiple difficulty levels and languages

## Key Features

- 18 supported languages: en, zh, ar, bn, de, es, fr, id, it, ja, ko, ms, pt, ru, sw, te, th, vi
- 4 difficulty levels: low, medium, high, top
- 9,000 high-quality problems total
- Language-specific instructions for each problem
- High-quality human translations ensuring accuracy

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with numeric comparison
- Additional metric: **DW-ACC** (Difficulty-Weighted Accuracy)
  - Weights: low=1, medium=2, high=4, top=8
  - Provides balanced scoring across difficulty levels
- Results reported per language and overall


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `poly_math` |
| **Dataset ID** | [evalscope/PolyMath](https://modelscope.cn/datasets/evalscope/PolyMath/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `MultiLingual`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 9,000 |
| Prompt Length (Mean) | 342.15 chars |
| Prompt Length (Min/Max) | 52 / 1536 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
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

## Sample Example

**Subset**: `en-low`

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

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets poly_math \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

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
            # subset_list: ['en-low', 'zh-low', 'ar-low']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


