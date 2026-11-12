# VLMs Are Biased


## 概述

VLMs Are Biased (VLMBias) 评估视觉语言模型是依据图像中的客观视觉信息回答问题，还是依赖记忆中的先验知识。该基准使用反事实图像，其可见属性与常见概念相冲突，例如带有四条条纹的 Adidas 风格标志或腿数异常的动物。

## 任务描述

- **任务类型**：用于计数和识别的自由格式视觉问答
- **输入**：一张反事实或对照图像，配有一个计数、二元识别或简答题
- **输出**：一个数字、`Yes`/`No`，或用花括号括起来的简短身份标识
- **领域**：动物、标志、旗帜、国际象棋棋子、棋盘、视错觉和图案网格

## 主要特点

- 主要的 `main` 子集包含 2,784 个客观视觉问题，基于 1,392 张分辨率为 384、768 和 1152 像素的反事实图像
- 五个官方分析子集涵盖二元识别、图像内标题注入、原始未修改对照以及背景移除变体
- 每条反事实记录同时提供视觉上正确的 `ground_truth` 和基于先验知识的 `expected_bias`
- 该基准暴露了七个主题和十九个子主题，以便进行详细分析，而无需创建合成的 EvalScope 子集
- 默认数据为 **v2**：视错觉目标特征为红色并在问题中命名，ID 全局唯一，且修复了截断的识别图像
- 两个 Hub 均固定到匹配的 v2 快照；样本元数据保留 v2 `ID`、原始 `legacy_ID` 以及选定的数据版本

## 评估说明

- 数据集提示词逐字使用，包括其要求的花括号答案格式
- 主要指标：**Accuracy** (`acc`)，在去除外层花括号后使用官方的不区分大小写比较；如果精确文本匹配失败，则比较数字序列
- 次要指标：**Bias Ratio** (`bias_ratio`，越低越好)，即在相同归一化条件下预测结果与 `expected_bias` 匹配的比例
- Accuracy 也按主题报告，以匹配官方的 lmms-eval 集成
- 对于 `original` 子集省略 `bias_ratio`，因为这些对照记录未定义 `expected_bias`
- 默认情况下，六个官方数据集子集作为单独的 EvalScope 子集进行评估；仅选择 `main` 进行主要的计数评估
- 默认的 v2 运行使用评估版本 **v1.1**。对于论文的 v1 数据，请显式设置 `dataset_args={'vlms_are_biased': {'default_subset': 'v1-legacy', 'subset_list': ['main']}}`；这些运行标记为 **v1 legacy** 并使用评估版本 **v1.0**。遗留的 `identification` 子集保留了上游截断图像的缺陷
- v1 和 v2 的结果不能直接互换。更改数据版本后请启动新的输出目录；不要使用 `rerun_review` 来复用针对已更改图像或提示词的预测结果。历史未固定的运行需要检查其保存的输入/ID 以确定数据版本
- 生成过程应具有确定性且简洁；官方 lmms-eval 设置使用 `temperature=0` 且最多生成 32 个新 token
- [论文](https://arxiv.org/abs/2505.23941) | [GitHub](https://github.com/anvo25/vlms-are-biased) | [项目页面](https://vlmsarebiased.github.io/)


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
| 提示词长度（平均） | 88.3 字符 |
| 提示词长度（最小/最大） | 60 / 130 字符 |

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
      "id": "0d71a23a",
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
