# ScreenSpot-Pro


## 概述

ScreenSpot-Pro 是一个 GUI 定位基准测试，基于专业桌面软件的真实高分辨率截图构建而成。给定一条自然语言指令，模型必须在屏幕上定位目标 UI 元素，这对在大型、密集布局的显示器上进行细粒度定位提出了挑战。

## 任务描述

- **任务类型**：GUI 定位（单击点预测）
- **输入**：一张全分辨率桌面截图 + 一段英文指令，用于描述目标 UI 元素
- **输出**：一个归一化到 [0, 1] 范围内的点击坐标 `[x, y]`，需在 `Answer:` 标记后给出
- **领域**：涵盖 CAD、创意设计、开发、办公、操作系统和科学计算等领域的专业桌面应用程序

## 主要特点

- 包含 1,581 条专家标注的指令，覆盖 26 款应用程序和 3 个平台（Windows、macOS、Linux）
- 截图均为真实高分辨率（最高达 6016x3384），目标元素通常仅占图像面积的 0.1% 以下
- 样本按六大专业领域分组（`CAD`、`Creative`、`Dev`、`OS`、`Office`、`Scientific`），每个领域作为一个子集提供
- 每个元素均标注为 `text`（文本）或 `icon`（图标），便于分别报告文本与图标目标的性能
- 真值边界框以像素坐标形式提供，并与原始图像尺寸配对，在评分前会进行归一化处理

## 评估说明

- 主要指标：**accuracy**（准确率）——当预测点落在真值边界框内时视为正确
- 辅助指标：**text_acc** 和 **icon_acc**，分别对对应 `ui_type` 的样本取平均
- 预测结果从提示要求的答案行中读取（`Answer: [x, y]`），因此推理过程不会被误认为答案。若回复忽略该格式，则仅尝试扫描明确的点表示法（如 `[x, y]` 对或 `<bbox>` 标签）；宽松写法（如 `x=.., y=..` 或单独数字）仅在答案行上被接受，因为在自由文本中这类写法可能捕获的是布局边界或序号而非点击坐标
- 若回复在答案行前被截断，则视为无预测，得分为 0（而非从推理中捏造坐标），因此请确保为模型预留足够的 `max_tokens` 以完成回答
- 真值已归一化至 [0, 1]，预测值根据其数值范围映射到同一空间：[0, 1] 视为归一化坐标；≤1000 的值视为许多视觉语言模型（VLM）常用的千分之一网格；更大数值则视为模型实际接收到的图像的像素坐标（所有截图宽度至少为 1920 像素，因此真实的像素坐标可被正确识别）
- 数据集仅提供一个 `train` 切分，用作评估切分
- 图像体积较大；可通过 `dataset_args` 中的 `max_image_bytes` 限制请求大小，像素空间的预测会根据实际发送图像的尺寸进行归一化
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
| **评估切分** | `train` |


## 数据统计

*统计数据不可用。*

## 样例示例

*样例示例不可用。*

## 提示模板

*未定义提示模板。*

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
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
