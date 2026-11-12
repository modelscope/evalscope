# VLMs Are Biased

## 概述

VLMs Are Biased (VLMBias) 测试视觉语言模型在视觉证据与常见概念冲突时（例如条纹数量异常的标志），是否遵循视觉证据。

## 任务描述

- **任务类型**：用于计数和识别的视觉问答
- **输入**：反事实或控制图像以及一个问题
- **输出**：一个数字、`Yes`/`No`，或用花括号括起来的简短答案
- **领域**：动物、标志、旗帜、国际象棋棋子、游戏棋盘、视错觉和图案网格

## 主要特点

- 六个子集中共有 11,594 个图像-问题对；`main` 包含 2,784 个问题
- 七个主题和十九个子主题，包括识别、标题注入、原始图像和背景移除分析
- 默认数据：**v2**，具有修订后的视错觉图像/问题、唯一 ID 以及修复后的识别图像

## 评估说明

- 指标：**准确率** (`acc`，总体及按主题) 和 **偏差比率** (`bias_ratio`，越低越好)，使用官方答案规范化和数字匹配；`original` 省略偏差比率
- 默认运行所有六个子集；设置 `subset_list=['main']` 以进行主要的计数评估
- 对于论文的 v1 数据，请在 `dataset_args` 中设置 `default_subset='v1-legacy'`。切换版本时使用新的输出目录。遗留的 `identification` 子集保留了一个截断的图像
- 提示词逐字使用。官方 lmms-eval 设置使用 `temperature=0` 且最多生成 32 个新 token
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
| **评估子集** | `main` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 11,594 |
| 提示词长度 (平均值) | 88.3 字符 |
| 提示词长度 (最小/最大) | 60 / 130 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `main` | 2,784 | 89.97 | 78 | 129 |
| `identification` | 1,392 | 83.27 | 68 | 102 |
| `withtitle` | 2,784 | 89.97 | 78 | 129 |
| `original` | 458 | 85.03 | 60 | 130 |
| `remove_background_q1q2` | 2,784 | 90.19 | 78 | 125 |
| `remove_background_q3` | 1,392 | 83.91 | 70 | 102 |

**图像统计：**

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
      "id": "a5b9d7f1",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~1.6KB]"
        },
        {
          "text": "Are the two red lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}."
        }
      ]
    }
  ],
  "target": "Yes",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "illusion-verticalhorizontal-001-diff0-notitle-q1-384",
    "legacy_id": "VerticalHorizontal_001_Q1_notitle_px384",
    "data_version": "v2",
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
