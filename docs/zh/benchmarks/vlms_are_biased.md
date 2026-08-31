# VLMs Are Biased


## 概述

VLMs Are Biased（VLMBias）用于评估视觉-语言模型（VLM）在回答客观视觉问题时，是依据图像内容作答，还是依赖于记忆中的先验知识。该基准测试使用反事实图像（counterfactual images），这些图像的可见属性与常见概念相冲突，例如带有四条纹的阿迪达斯风格标志，或拥有异常腿数的动物。

## 任务描述

- **任务类型**：自由形式的视觉问答（计数与识别）
- **输入**：一张反事实图像或对照图像，配以计数、二元识别或简短回答类问题
- **输出**：一个数字、`Yes`/`No`，或用花括号括起的简短身份标识
- **领域**：动物、商标、旗帜、国际象棋棋子、游戏棋盘、视错觉和图案网格

## 核心特性

- 主要的 `main` 划分包含 2,784 个客观视觉问题，覆盖 1,392 张反事实图像，分辨率分别为 384、768 和 1152 像素
- 五个官方分析划分涵盖二元识别、图像内标题注入、原始未修改对照图像，以及去除背景的变体
- 每条反事实记录同时提供视觉上正确的 `ground_truth` 和基于先验知识的 `expected_bias`
- 该基准测试涵盖七个主题和十九个子主题，支持细粒度分析，无需创建合成的 EvalScope 子集

## 评估说明

- 数据集提示词按原文使用，包括其要求的花括号答案格式
- 主要指标：**准确率**（`acc`），采用官方提供的大小写不敏感比较方法（去除外层花括号后）；若精确文本匹配失败，则比较其中的数字序列
- 次要指标：**偏见比率**（`bias_ratio`，越低越好），即在相同归一化条件下，预测结果与 `expected_bias` 一致的比例
- 准确率也按主题分别报告，与官方 lmms-eval 集成一致
- `original` 划分不计算 `bias_ratio`，因为这些对照样本未定义 `expected_bias`
- 六个官方数据集划分作为独立的 EvalScope 子集公开，并默认全部参与评估；若仅选择 `main` 子集，可复现论文中的主要基准结果
- 生成应具有确定性且简洁；官方 lmms-eval 设置使用 `temperature=0`，最多生成 32 个新 token
- [论文](https://arxiv.org/abs/2505.23941) | [GitHub](https://github.com/anvo25/vlms-are-biased) | [项目主页](https://vlmsarebiased.github.io/)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `vlms_are_biased` |
| **数据集ID** | [evalscope/vlms-are-biased](https://modelscope.cn/datasets/evalscope/vlms-are-biased/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2505.23941) |
| **标签** | `MultiModal`, `QA`, `Reasoning` |
| **指标** | `accuracy`, `bias_ratio` |
| **默认示例数** | 0-shot |
| **评估划分** | `main` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 11,594 |
| 提示词长度（平均） | 90.01 字符 |
| 提示词长度（最小/最大） | 60 / 138 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `main` | 2,784 | 91.52 | 78 | 129 |
| `identification` | 1,392 | 83.27 | 68 | 102 |
| `withtitle` | 2,784 | 91.52 | 78 | 129 |
| `original` | 458 | 85.03 | 60 | 130 |
| `remove_background_q1q2` | 2,784 | 94.23 | 78 | 138 |
| `remove_background_q3` | 1,392 | 83.91 | 70 | 102 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 11,594 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 384x183 - 1862x1430 |
| 格式 | png |


## 样例示例

**子集**: `main`

```json
{
  "input": [
    {
      "id": "3872fe66",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~1.9KB]"
        },
        {
          "text": "Are the horizontal and vertical lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}."
        }
      ]
    }
  ],
  "target": "Yes",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "VerticalHorizontal_001_Q1_notitle_px384",
    "topic": "Optical Illusion",
    "sub_topic": "Vertical-Horizontal illusion",
    "type_of_question": "Q1",
    "expected_bias": "No",
    "with_title": false,
    "pixel": 384
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
    --datasets vlms_are_biased \
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
    datasets=['vlms_are_biased'],
    dataset_args={
        'vlms_are_biased': {
            # subset_list: ['main', 'identification', 'withtitle']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
