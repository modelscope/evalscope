# MMBench


## 概述

MMBench 是一个系统性设计的基准测试，用于在 20 个细粒度能力维度上评估视觉-语言模型。它采用了一种新颖的 CircularEval 策略，并提供英文和中文两个版本，以支持跨语言评估。

## 任务描述

- **任务类型**：视觉多选问答（Visual Multiple-Choice Q&A）
- **输入**：包含问题和 2–4 个选项的图像
- **输出**：单个正确答案字母（A、B、C 或 D）
- **语言**：包含英文（en）和中文（cn）子集

## 主要特性

- 定义了 20 个细粒度能力维度
- 采用 CircularEval 策略的系统化评估流程
- 支持双语（英文和中文）
- 问题中包含上下文提示（hints）
- 考察感知、推理和知识能力

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用思维链（Chain-of-Thought, CoT）提示
- 在开发集（dev split）上进行评估
- 包含两个子集：`cn`（中文）和 `en`（英文）
- 结果包含按类别划分的详细指标

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mm_bench` |
| **数据集 ID** | [lmms-lab/MMBench](https://modelscope.cn/datasets/lmms-lab/MMBench/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `dev` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 8,658 |
| 提示词长度（平均） | 372.3 字符 |
| 提示词长度（最小/最大） | 241 / 2395 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `cn` | 4,329 | 312.28 | 241 | 1681 |
| `en` | 4,329 | 432.33 | 256 | 2395 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 8,658 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 106x56 - 512x512 |
| 格式 | jpeg |


## 样例示例

**子集**: `cn`

```json
{
  "input": [
    {
      "id": "00b49734",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B. Think step by step before answering.\n\n下面的文章描述了一个实验。阅读文章，然后按照以下说明进行操作。\n\nMadelyn在雪板的底部涂上了一层薄蜡，然后直接下坡滑行。然后，她去掉了蜡，再次直接下坡滑行。她重复了这个过程四次，每次都交替使用薄蜡或不使用薄蜡滑行。她的朋友Tucker计时每次滑行的时间。Madelyn和Tucker计算了使用薄蜡滑行和不使用薄蜡滑行时直接下坡所需的平均时间。\n图：滑雪板下坡。麦德琳和塔克的实验能最好回答哪个问题？\n\nA) 当麦德琳的雪板上有一层薄蜡或一层厚蜡时，它是否能在较短的时间内滑下山坡？\nB) 当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？"
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~11.7KB]"
        }
      ]
    }
  ],
  "choices": [
    "当麦德琳的雪板上有一层薄蜡或一层厚蜡时，它是否能在较短的时间内滑下山坡？",
    "当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 241,
    "category": "identity_reasoning",
    "source": "scienceqa",
    "L2-category": "attribute_reasoning",
    "comment": "nan",
    "split": "dev"
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
    --datasets mm_bench \
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
    datasets=['mm_bench'],
    dataset_args={
        'mm_bench': {
            # subset_list: ['cn', 'en']  # 可选，用于指定评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```