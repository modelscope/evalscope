# MMMU-PRO

## 概述

MMMU-PRO 是一个增强版的多模态基准测试，旨在严格评估先进 AI 模型在多种模态下的真实理解能力。它在原始 MMMU 基准的基础上进行了关键改进，使评估更具挑战性和现实性。

## 任务描述

- **任务类型**：多模态学术问答
- **输入**：图像（最多 7 张）+ 多选题
- **输出**：正确答案选项字母
- **领域**：涵盖 STEM、人文和社会科学领域的 30 个学科

## 主要特性

- MMMU 的增强版本，提供更严格的评估
- 覆盖 30 个学科：会计学、生物学、化学、计算机科学、经济学、物理学等
- 提供多种数据集格式：
  - `standard (4 options)`：传统的 4 选项格式
  - `standard (10 options)`：扩展的 10 选项格式，用于更难的评估
  - `vision`：问题嵌入在图像中
- 测试真实的多模态理解能力，而非仅依赖文本捷径

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：多选题的 **准确率（Accuracy）**
- 可通过 `dataset_format` 参数配置数据集格式
- 使用思维链（Chain-of-Thought, CoT）提示进行推理
- 包含丰富的元数据，如题目难度和学科信息

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmmu_pro` |
| **数据集 ID** | [AI-ModelScope/MMMU_Pro](https://modelscope.cn/datasets/AI-ModelScope/MMMU_Pro/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,730 |
| 提示词长度（平均） | 521.89 字符 |
| 提示词长度（最小/最大） | 249 / 3749 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Accounting` | 58 | 518.48 | 320 | 899 |
| `Agriculture` | 60 | 477.05 | 289 | 747 |
| `Architecture_and_Engineering` | 60 | 592.85 | 281 | 1177 |
| `Art` | 53 | 358.34 | 297 | 919 |
| `Art_Theory` | 55 | 362.53 | 289 | 619 |
| `Basic_Medical_Science` | 52 | 401.96 | 277 | 867 |
| `Biology` | 59 | 476.81 | 269 | 1387 |
| `Chemistry` | 60 | 453.75 | 264 | 1217 |
| `Clinical_Medicine` | 59 | 525.58 | 311 | 977 |
| `Computer_Science` | 60 | 441.37 | 262 | 1077 |
| `Design` | 60 | 408.25 | 285 | 1449 |
| `Diagnostics_and_Laboratory_Medicine` | 60 | 444.17 | 274 | 789 |
| `Economics` | 59 | 506.37 | 284 | 900 |
| `Electronics` | 60 | 455.6 | 314 | 668 |
| `Energy_and_Power` | 58 | 506.86 | 347 | 816 |
| `Finance` | 60 | 637.75 | 317 | 1864 |
| `Geography` | 52 | 409.9 | 267 | 929 |
| `History` | 56 | 611.3 | 328 | 1077 |
| `Literature` | 52 | 429.87 | 274 | 564 |
| `Manage` | 50 | 666.56 | 282 | 2198 |
| `Marketing` | 59 | 596.53 | 303 | 1060 |
| `Materials` | 60 | 484.02 | 296 | 1351 |
| `Math` | 60 | 511.95 | 249 | 1172 |
| `Mechanical_Engineering` | 59 | 527.95 | 272 | 1418 |
| `Music` | 60 | 336.55 | 250 | 672 |
| `Pharmacy` | 57 | 474.7 | 282 | 902 |
| `Physics` | 60 | 499.07 | 341 | 737 |
| `Psychology` | 60 | 1355.12 | 280 | 3749 |
| `Public_Health` | 58 | 716.98 | 282 | 2510 |
| `Sociology` | 54 | 416.48 | 279 | 708 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 2,048 |
| 每样本图像数 | 最小: 1, 最大: 35, 平均: 1.18 |
| 分辨率范围 | 43x50 - 2560x2545 |
| 格式 | png |

## 样例示例

**子集**: `Accounting`

```json
{
  "input": [
    {
      "id": "bae49033",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C. Think step by step before answering.\n\nPrices of zero-coupon bonds reveal the following pattern of forward rates: "
        },
        {
          "image": "[BASE64_IMAGE: png, ~8.0KB]"
        },
        {
          "text": " In addition to the zero-coupon bond, investors also may purchase a 3-year bond making annual payments of $60 with par value $1,000. Under the expectations hypothesis, what is the expected realized compound yield of the coupon bond?\n\nA) 6.66%\nB) 6.79%\nC) 6.91%"
        }
      ]
    }
  ],
  "choices": [
    "6.66%",
    "6.79%",
    "6.91%"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "subset_key": "Accounting",
  "metadata": {
    "id": "test_Accounting_42",
    "explanation": "?",
    "img_type": "['Tables']",
    "topic_difficulty": "Hard",
    "subject": "Accounting"
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

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `dataset_format` | `str` | `standard (4 options)` | 数据集格式变体。可选值：['standard (4 options)', 'standard (10 options)', 'vision'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmmu_pro \
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
    datasets=['mmmu_pro'],
    dataset_args={
        'mmmu_pro': {
            # subset_list: ['Accounting', 'Agriculture', 'Architecture_and_Engineering']  # 可选，评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```