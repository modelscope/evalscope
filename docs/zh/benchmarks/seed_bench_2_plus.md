# SEED-Bench-2-Plus

## 概述

SEED-Bench-2-Plus 是一个大规模基准测试，旨在评估多模态大语言模型（MLLMs）在富含文本的视觉理解任务上的表现。该基准包含 2.3K 道多项选择题，覆盖真实世界场景，并配有精确的人工标注。

## 任务描述

- **任务类型**：富含文本的视觉问答（Text-Rich Visual Question Answering）
- **输入**：包含丰富文本内容的图像 + 多项选择题
- **输出**：正确答案选项字母（A/B/C/D）
- **领域**：图表（Charts）、地图（Maps）、网页界面（Web Interfaces）

## 主要特点

- 聚焦于真实应用场景中常见的富含文本的视觉场景
- 包含三大类别：**图表（Charts）**、**地图（Maps）** 和 **网页（Webs）**
- 问题由人工高质量标注
- 测试模型对复杂图文布局的理解能力
- 每个类别内包含多个难度级别

## 评估说明

- 默认使用 **test** 数据集划分进行评估
- 可用子集：`chart`、`web`、`map`
- 主要指标：多项选择题的 **准确率（Accuracy）**
- 使用思维链（Chain-of-Thought, CoT）提示进行推理
- 提供丰富的元数据，包括数据来源、类型和难度等级

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `seed_bench_2_plus` |
| **数据集ID** | [evalscope/SEED-Bench-2-Plus](https://modelscope.cn/datasets/evalscope/SEED-Bench-2-Plus/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,277 |
| 提示词长度（平均） | 393.01 字符 |
| 提示词长度（最小/最大） | 280 / 710 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `chart` | 810 | 390.92 | 280 | 663 |
| `web` | 660 | 395.56 | 294 | 664 |
| `map` | 807 | 393.02 | 282 | 710 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 2,277 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 800x800 - 800x800 |
| 格式 | png |

## 样例示例

**子集**: `chart`

```json
{
  "input": [
    {
      "id": "7fdf638f",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nAccording to the tree diagram, how many women who have breast cancer received a negative mammogram?\n\nA) 80\nB) 950\nC) 20\nD) 8,950"
        },
        {
          "image": "[BASE64_IMAGE: png, ~68.1KB]"
        }
      ]
    }
  ],
  "choices": [
    "80",
    "950",
    "20",
    "8,950"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "chart",
  "metadata": {
    "data_id": "text_rich/1.png",
    "question_id": "0",
    "question_image_subtype": "tree diagram",
    "data_source": "SEED-Bench v2 plus",
    "data_type": "Single Image",
    "level": "L1",
    "subpart": "Single-Image & Text Comprehension",
    "version": "v2+"
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
    --datasets seed_bench_2_plus \
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
    datasets=['seed_bench_2_plus'],
    dataset_args={
        'seed_bench_2_plus': {
            # subset_list: ['chart', 'web', 'map']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```