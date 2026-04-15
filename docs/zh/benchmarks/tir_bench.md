# TIR-Bench


## 概述

TIR-Bench（Thinking-with-Images Reasoning Benchmark）是一个全面的多模态基准测试，用于评估视觉语言模型的具身视觉推理能力。该基准涵盖多种任务类别，要求模型具备空间、组合式及多步骤的视觉推理能力。

## 任务描述

- **任务类型**：多任务视觉推理（选择题、OCR、单词搜索、找不同、拼图等）
- **输入**：一张或两张图像 + 问题（大多数任务采用选择题格式）
- **输出**：答案字母（选择题）或数字/文本形式的答案（取决于任务类型）
- **领域**：instrument（仪器）、color（颜色）、refcoco、rotation_game（旋转游戏）、math（数学）、word_search（单词搜索）、visual_search（视觉搜索）、ocr、symbolic（符号）、spot_difference（找不同）、contrast（对比）、jigsaw（拼图）、maze（迷宫）

## 核心特性

- 包含13种多样化的视觉推理任务类别，共计1,215个测试样本
- 覆盖单图和双图推理场景
- 答案形式包括字母选项（A-J）、整数、浮点数和文本
- 采用任务特定评分机制，并辅以LLM-as-judge作为后备方案，确保评估的鲁棒性

## 评估说明

- 默认使用 **test** 划分进行评估（共1,215个样本）
- 主要指标：**准确率**（acc）
- 图像数据通过 ModelScope 下载为 `data.zip` 并自动解压
- 基于规则的评分方式：
  - OCR（子串匹配）
  - 拼图（grid IoU）
  - 找不同（set IoU）
  - 单词搜索（数值匹配）
  - 其他所有任务（选择题 / 数值判断）
- **推荐设置**：将 `judge_strategy=JudgeStrategy.LLM_RECALL` 并提供 `judge_model_args`，以启用 LLM-as-judge 作为召回机制——仅当基于规则的评分结果为0时才调用判别模型，在避免不必要 API 开销的同时提升评估准确性
- [论文](https://arxiv.org/abs/2511.01833) | [GitHub](https://github.com/agents-x-project/TIR-Bench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `tir_bench` |
| **数据集ID** | [evalscope/TIR-Bench](https://modelscope.cn/datasets/evalscope/TIR-Bench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2511.01833) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,215 |
| 提示词长度（平均） | 384.97 字符 |
| 提示词长度（最小/最大） | 19 / 4039 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `instrument` | 80 | 110.96 | 57 | 196 |
| `color` | 100 | 130.26 | 98 | 241 |
| `refcoco` | 120 | 144.51 | 132 | 182 |
| `rotation_game` | 75 | 146.44 | 140 | 148 |
| `math` | 120 | 126.69 | 50 | 397 |
| `word_search` | 100 | 126.72 | 24 | 307 |
| `visual_search` | 120 | 111.64 | 19 | 501 |
| `ocr` | 60 | 35.08 | 29 | 116 |
| `symbolic` | 50 | 88.18 | 66 | 243 |
| `spot_difference` | 100 | 1114.79 | 93 | 1379 |
| `contrast` | 50 | 48.1 | 31 | 123 |
| `jigsaw` | 120 | 605 | 605 | 605 |
| `maze` | 120 | 1527.06 | 626 | 4039 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 1,255 |
| 每样本图像数 | 最小: 1, 最大: 2, 平均: 1.03 |
| 分辨率范围 | 60x23 - 6944x9280 |
| 格式 | jpeg, mpo, png, webp |


## 样例示例

**子集**: `instrument`

```json
{
  "input": [
    {
      "id": "dd25f10d",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpg, ~2.9MB]"
        },
        {
          "text": "According to the image, what is the thermometer reading in Fahrenheit? Answer as an integer like 1,2,3."
        }
      ]
    }
  ],
  "target": "72",
  "id": 0,
  "group_id": 0,
  "subset_key": "instrument",
  "metadata": {
    "task": "instrument",
    "meta_data": {},
    "id": 6
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets tir_bench \
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
    datasets=['tir_bench'],
    dataset_args={
        'tir_bench': {
            # subset_list: ['instrument', 'color', 'refcoco']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```