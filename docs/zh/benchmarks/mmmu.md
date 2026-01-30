# MMMU


## 概述

MMMU（Massive Multi-discipline Multimodal Understanding，大规模多学科多模态理解）是一个综合性基准测试，旨在评估多模态模型在需要大学水平学科知识和深度推理的专家级任务上的表现。该基准涵盖6个核心学科领域的30个具体科目。

## 任务描述

- **任务类型**：多模态问答（选择题与开放式问题）
- **输入**：包含多样化图像（图表、示意图、地图、表格等）的问题
- **输出**：答案选项字母（选择题）或自由文本（开放式问题）
- **学科领域**：艺术与设计、商业、科学、健康与医学、人文学科、技术与工程

## 核心特性

- 精心收集的11.5K个多模态问题
- 数据来源包括大学考试、测验和教材
- 覆盖30个科目和183个子领域
- 包含30种异构图像类型（图表、示意图、乐谱、化学结构式等）
- 同时考察感知能力和专家级推理能力

## 评估说明

- 默认配置采用 **0-shot** 评估方式
- 支持选择题和开放式问题两种题型
- 单个问题最多可包含7张图像
- 对于开放式问题，需按 "ANSWER: [ANSWER]" 格式输出答案
- 评估使用验证集（测试集需提交结果）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmmu` |
| **数据集ID** | [AI-ModelScope/MMMU](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 900 |
| 提示词长度（平均） | 527.84 字符 |
| 提示词长度（最小/最大） | 247 / 3011 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Accounting` | 30 | 525.2 | 356 | 899 |
| `Agriculture` | 30 | 472.73 | 321 | 747 |
| `Architecture_and_Engineering` | 30 | 643.33 | 279 | 927 |
| `Art` | 30 | 384.17 | 297 | 1098 |
| `Art_Theory` | 30 | 370.13 | 297 | 588 |
| `Basic_Medical_Science` | 30 | 420.1 | 277 | 1119 |
| `Biology` | 30 | 524.87 | 294 | 1239 |
| `Chemistry` | 30 | 522.27 | 286 | 1220 |
| `Clinical_Medicine` | 30 | 549.37 | 311 | 914 |
| `Computer_Science` | 30 | 512.73 | 273 | 1285 |
| `Design` | 30 | 401.77 | 292 | 613 |
| `Diagnostics_and_Laboratory_Medicine` | 30 | 440.73 | 302 | 741 |
| `Economics` | 30 | 503.63 | 354 | 730 |
| `Electronics` | 30 | 476.83 | 314 | 659 |
| `Energy_and_Power` | 30 | 529.17 | 347 | 814 |
| `Finance` | 30 | 628.67 | 361 | 985 |
| `Geography` | 30 | 460.37 | 299 | 759 |
| `History` | 30 | 590.63 | 360 | 899 |
| `Literature` | 30 | 425.57 | 275 | 541 |
| `Manage` | 30 | 784 | 414 | 2261 |
| `Marketing` | 30 | 530.6 | 303 | 832 |
| `Materials` | 30 | 474.63 | 308 | 667 |
| `Math` | 30 | 500.6 | 247 | 1167 |
| `Mechanical_Engineering` | 30 | 504.73 | 272 | 861 |
| `Music` | 30 | 315.93 | 250 | 455 |
| `Pharmacy` | 30 | 499.4 | 311 | 981 |
| `Physics` | 30 | 525.83 | 332 | 1086 |
| `Psychology` | 30 | 1173.9 | 280 | 3011 |
| `Public_Health` | 30 | 678.17 | 282 | 2684 |
| `Sociology` | 30 | 465.27 | 299 | 1007 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 980 |
| 每样本图像数 | 最小: 1, 最大: 5, 平均: 1.09 |
| 分辨率范围 | 70x67 - 2560x2133 |
| 格式 | png |


## 样例示例

**子集**: `Accounting`

```json
{
  "input": [
    {
      "id": "be505d26",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~65.8KB]"
        },
        {
          "text": " Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?\n\nA) $6\nB) $7\nC) $8\nD) $9"
        }
      ]
    }
  ],
  "choices": [
    "$6",
    "$7",
    "$8",
    "$9"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "validation_Accounting_1",
    "question_type": "multiple-choice",
    "subfield": "Managerial Accounting",
    "explanation": "",
    "img_type": "['Tables']",
    "topic_difficulty": "Medium"
  }
}
```

## 提示模板

**提示模板：**
```text

Solve the following problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \boxed command.


```

## 使用方法

### 使用命令行

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmmu \
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
    datasets=['mmmu'],
    dataset_args={
        'mmmu': {
            # subset_list: ['Accounting', 'Agriculture', 'Architecture_and_Engineering']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```