# ScreenSpot-Pro


## 概述

ScreenSpot-Pro 是一个 GUI 定位基准测试，基于专业桌面软件的真实高分辨率截图构建而成。给定一条自然语言指令，模型必须在屏幕上定位目标 UI 元素，这对在大型、密集布局的显示器上进行细粒度定位提出了挑战。

## 任务描述

- **任务类型**：GUI 定位（单击点预测）
- **输入**：一张全分辨率桌面截图 + 一条描述目标 UI 元素的英文指令
- **输出**：一个归一化坐标范围 [0, 1] 内的点击点 `[x, y]`
- **领域**：涵盖 CAD、创意设计、开发、办公、操作系统和科学计算等专业桌面应用

## 核心特性

- 包含 1,581 条专家标注的指令，覆盖 26 款应用及 3 个平台（Windows、macOS、Linux）
- 截图均为真实高分辨率（最高达 6016x3384），目标元素通常仅占图像面积的 0.1% 以下
- 样本按六大专业领域分组（`CAD`、`Creative`、`Dev`、`OS`、`Office`、`Scientific`），每个领域作为一个子集提供
- 每个元素均标注为 `text`（文本）或 `icon`（图标），便于分别报告文本与图标类目标的性能
- 真值框以像素坐标形式提供，并与原始图像尺寸配对，在评分前会进行归一化处理

## 评估说明

- 主要指标：**acc** — 当预测点落在真值边界框内时视为正确
- 次要指标：**text_acc** 和 **icon_acc**，分别对对应 `ui_type` 的样本取平均
- 预测结果可从 `[x, y]` 点、`x=.., y=..` 对或边界框（取其中心点）中解析，始终采用*最后一个匹配项*，以避免重述提示或逐步推理干扰最终答案
- 真值已归一化至 [0, 1] 范围，预测值会根据 `coordinate_space` 映射到相同空间。默认 `auto` 模式下：[0, 1] 范围内的值视为已归一化；≤1000 的值视为许多视觉语言模型（VLMs）常用的千分网格；更大数值则视为模型接收图像的像素坐标。若已知模型的坐标约定，请显式指定 `coordinate_space`，因为 1–1000 范围本身存在歧义
- 数据集仅提供一个 `train` 划分，用作评估划分
- 图像体积较大；可通过 `dataset_args` 中的 `max_image_bytes` 限制请求大小，像素空间的预测会根据实际发送图像的尺寸进行归一化
- [论文](https://arxiv.org/abs/2504.07981) | [GitHub](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `screenspot_pro` |
| **数据集ID** | [lmms-lab/ScreenSpot-Pro](https://modelscope.cn/datasets/lmms-lab/ScreenSpot-Pro/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2504.07981) |
| **标签** | `Agent`, `Grounding`, `MultiModal` |
| **指标** | `acc`, `text_acc`, `icon_acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,581 |
| 提示词长度（平均） | 199.22 字符 |
| 提示词长度（最小/最大） | 175 / 275 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `CAD` | 261 | 193.18 | 176 | 224 |
| `Creative` | 341 | 198.02 | 176 | 275 |
| `Dev` | 299 | 209.79 | 176 | 272 |
| `OS` | 196 | 197.57 | 177 | 262 |
| `Office` | 230 | 200.76 | 176 | 252 |
| `Scientific` | 254 | 194.44 | 175 | 233 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,581 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 1920x1080 - 6016x3384 |
| 格式 | png |


## 样例示例

**子集**: `CAD`

```json
{
  "input": [
    {
      "id": "83ef9e39",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~933.3KB]"
        },
        {
          "text": "Identify the UI element for the instruction and output exactly one click point as [x, y] in normalized coordinates within [0, 1]. Do not output a bounding box.\nInstruction: Mark dimensions"
        }
      ]
    }
  ],
  "target": "[0.1672, 0.0435, 0.1802, 0.1019]",
  "id": 0,
  "group_id": 0,
  "subset_key": "CAD",
  "metadata": {
    "id": "inventor_windows_0",
    "sent_size": [
      3840,
      1080
    ],
    "bbox_norm": [
      0.1671875,
      0.04351851851851852,
      0.18020833333333333,
      0.10185185185185185
    ],
    "ui_type": "text",
    "application": "inventor",
    "platform": "windows"
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `coordinate_space` | `str` | `auto` | 用于解释预测点击点的坐标约定。auto: 根据坐标数值大小自动推断；normalized: 值已在 [0, 1] 范围内；thousandths: 值位于 0-1000 网格上；pixel: 模型接收图像的像素坐标。可选值: ['auto', 'normalized', 'thousandths', 'pixel'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets screenspot_pro \
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
    datasets=['screenspot_pro'],
    dataset_args={
        'screenspot_pro': {
            # subset_list: ['CAD', 'Creative', 'Dev']  # 可选，评估特定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
