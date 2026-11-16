# ScreenSpot-Pro


## 概述

ScreenSpot-Pro 是一个基于专业桌面软件真实高分辨率截图构建的 GUI 定位基准测试。给定自然语言指令，模型必须在屏幕上定位目标 UI 元素，这考验了在大尺寸、高密度显示界面上的细粒度定位能力。

## 任务描述

- **任务类型**: GUI 定位（单点击点预测）
- **输入**: 全分辨率桌面截图 + 描述目标 UI 元素的英文指令
- **输出**: 一个归一化到 0 到 1 范围的点击点 `[x, y]`，位于 `Answer:` 标记之后
- **领域**: 涵盖 CAD、创意、开发、办公、操作系统和科学软件的专业桌面应用程序

## 主要特点

- 包含 1,581 条专家标注的指令，覆盖 26 个应用程序和 3 个平台（Windows, macOS, Linux）
- 截图具有真正的高分辨率（高达 6016x3384），因此目标元素通常占据图像面积的不到 0.1%
- 样本被分为六个专业领域 (`CAD`, `Creative`, `Dev`, `OS`, `Office`, `Scientific`)，每个领域作为一个子集公开
- 每个元素都标记为 `text` 或 `icon`，从而可以分别报告文本目标和图标目标的性能
- 真值框是像素坐标并配有原始图像尺寸，在评分前进行归一化处理

## 评估说明

- 主要指标: **accuracy** — 当预测点落在真值边界框内时，视为预测正确
- 次要指标: **text_acc** 和 **icon_acc**，分别对相应 `ui_type` 的样本取平均值
- 预测结果从提示词要求的回答行读取（`Answer: [x, y]`），因此推理轨迹不会被误认为是答案。忽略格式的回答仅回退到扫描无歧义的点表示法（`[x, y]` 对或 `<bbox>` 标签）；像 `x=.., y=..` 这样的松散表示法和裸数字仅在回答行中被接受，因为在自由文本中它们会捕获布局边界和序数而不是点击点
- 如果在回答行之前回复被截断，则不会产生预测且得分为 0，而不是从其推理中编造坐标，因此请允许足够的 `max_tokens` 以便模型完成回答
- 真值归一化为 [0, 1]，预测映射到同一空间。默认情况下（`extra_params` 中的 `coordinate_format: auto`），通过数值大小推断约定：[0, 1] 范围内的值被视为归一化值，高达 1000 的值被视为许多 VLM 输出的千分位网格，更大的值被视为模型接收到的图像的像素。因此，两个坐标均不超过 1000 的像素答案（即截图左上角 1000x1000 px 区域）会被读作千分位；对于具有固定约定的模型，请将 `coordinate_format` 设置为 `normalized`、`thousandths` 或 `pixel`
- 数据集仅提供一个 `train` 分割，该分割用作评估分割
- 图像较大；`dataset_args` 中的 `max_image_bytes` 可以限制请求大小，并且像素空间的预测使用实际发送的图像尺寸进行归一化。如果该图像的两边均小于或等于 1000 px，`auto` 会将所有像素答案读作千分位，因此对于以像素形式回答的模型，请设置 `coordinate_format: pixel`
- [论文](https://arxiv.org/abs/2504.07981) | [GitHub](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `screenspot_pro` |
| **数据集ID** | [lmms-lab/ScreenSpot-Pro](https://modelscope.cn/datasets/lmms-lab/ScreenSpot-Pro/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2504.07981) |
| **标签** | `Agent`, `Grounding`, `MultiModal` |
| **指标** | `accuracy`, `text_acc`, `icon_acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,581 |
| 提示词长度 (平均) | 319.22 字符 |
| 提示词长度 (最小/最大) | 295 / 395 字符 |

**各子集统计:**

| 子集 | 样本数 | 提示词均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `CAD` | 261 | 313.18 | 296 | 344 |
| `Creative` | 341 | 318.02 | 296 | 395 |
| `Dev` | 299 | 329.79 | 296 | 392 |
| `OS` | 196 | 317.57 | 297 | 382 |
| `Office` | 230 | 320.76 | 296 | 372 |
| `Scientific` | 254 | 314.44 | 295 | 353 |

**图像统计:**

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
      "id": "3c262e7c",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~933.3KB]"
        },
        {
          "text": "Identify the UI element for the instruction and give a single click point. Coordinates must be normalized to the range 0 to 1 relative to the image size. Do not output a bounding box.\nInstruction: Mark dimensions\nEnd your reply with the final answer on its own last line, formatted exactly as: Answer: [x, y]"
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
| `coordinate_format` | `str` | `auto` | 预测点击点的约定："auto" 根据数值大小推断；"normalized" (0-1)、"thousandths" (0-1000) 或 "pixel" (发送图像的像素) 固定约定。选项: ['auto', 'normalized', 'thousandths', 'pixel'] |

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
            # subset_list: ['CAD', 'Creative', 'Dev']  # 可选，用于评估特定子集
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
