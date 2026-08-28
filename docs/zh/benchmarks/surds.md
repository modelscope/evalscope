# SURDS


## 概述

SURDS 通过在真实驾驶场景中对视觉-语言模型进行评测，衡量其细粒度的空间理解与推理能力。该基准源自六摄像头 nuScenes 数据集，在不提供深度图或视觉标记的情况下，评估以物体为中心及关系型的空间技能。

## 任务描述

- **任务类型**：多任务视觉空间问答
- **输入**：一张 1600 x 900 的驾驶场景图像和一个英文空间推理问题
- **输出**：结构化响应，答案需包含在 `<answer>...</answer>` 标签内
- **领域**：自动驾驶与户外 3D 空间推理

## 关键特性

- 基于官方 seed-42 代码，从 5,919 张验证图像中确定性地生成了 9,250 条模型查询
- 包含六个权重相等的任务子集：偏航角（yaw orientation）、像素定位（pixel localization）、深度范围（depth range）、成对距离（pairwise distance）、左右顺序（left/right ordering）以及前后关系（front/behind relation）
- 偏航角、距离、左右、前后四项任务为一致性测试：每个评测单元的两个互补提示都必须正确才能得分
- 图像来自 nuScenes 的六个摄像头，其中包含未加标记的物体，仅通过外观描述而非叠加标注

## 评测说明

- 官方提示词及 `<think>...<answer>...</answer>` 响应格式被原样复现
- 像素定位采用官方的中心度（centerness）指标：预测点若在目标框外得分为 0；越接近框中心，得分越趋近于 1；也接受归一化坐标和预测框形式
- 其余五项任务使用官方的归一化精确匹配（normalized exact match），忽略大小写、标点、冠词及多余空格
- 每个子集包含 925 个评测单元；总体归一化分数为六个子集分数的等权平均。完整运行需发起 9,250 次模型请求，但报告的 `Num=5,550`，因为每对互补提示被视为一个官方评测单元
- 无效或缺失 `<answer>` 块的响应得分为 0，符合官方基准的分母语义
- 该数据集仅用于评测，从 ModelScope 下载；仅获取所选子集所需的图像
- 资源链接：[论文](https://arxiv.org/abs/2411.13112) |
  [GitHub](https://github.com/XiandaGuo/Drive-MLLM)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `surds` |
| **数据集ID** | [evalscope/SURDS_eval](https://modelscope.cn/datasets/evalscope/SURDS_eval/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2411.13112) |
| **标签** | `Grounding`, `MultiModal`, `QA`, `Reasoning` |
| **指标** | `normalized_score` |
| **默认示例数** | 0-shot |
| **评测划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 9,250 |
| 提示词长度（平均） | 728.31 字符 |
| 提示词长度（最小/最大） | 631 / 910 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `yaw` | 1,850 | 725.18 | 714 | 769 |
| `xy2d` | 925 | 677.32 | 672 | 717 |
| `depth` | 925 | 861.75 | 854 | 901 |
| `distance` | 1,850 | 770.04 | 743 | 910 |
| `left_right` | 1,850 | 658.04 | 631 | 798 |
| `front_behind` | 1,850 | 718.77 | 703 | 791 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 9,250 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 1600x900 - 1600x900 |
| 格式 | webp |


## 样例示例

**子集**: `yaw`

```json
{
  "input": [
    {
      "id": "31f24a57",
      "content": [
        {
          "text": "Task Description: \nThe primary goal of this task is to identify the direction that the specified object is facing in the given image. The camera in the image is facing North, and you need to analyze the object's orientation based on this refe ... [TRUNCATED 232 chars] ... evant error checks.\nFinally, provide a concise and definitive response in the <answer> tag. Use the following format:\n<think>[Step-by-step reasoning with attention to detail and potential error checks]</think>\n<answer>[Final answer]</answer>\n"
        },
        {
          "image": "~/.cache/modelscope/hub/datasets/evalscope/SURDS_eval/validation/image/CAM_BACK_RIGHT/nuscenes_0033_CAM_BACK_RIGHT.webp"
        }
      ]
    }
  ],
  "target": "West",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task": "yaw",
    "pair_id": "yaw-3",
    "variant_index": 0,
    "paired": true,
    "bbox": [
      662,
      504,
      774,
      545
    ],
    "options": [
      "North",
      "South",
      "East",
      "West"
    ],
    "image_size": [
      1600,
      900
    ],
    "image_path": "~/.cache/modelscope/hub/datasets/evalscope/SURDS_eval/validation/image/CAM_BACK_RIGHT/nuscenes_0033_CAM_BACK_RIGHT.webp"
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets surds \
    --limit 10  # 正式评测时请删除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['surds'],
    dataset_args={
        'surds': {
            # subset_list: ['yaw', 'xy2d', 'depth']  # 可选，用于评测特定子集
        }
    },
    limit=10,  # 正式评测时请删除此行
)

run_task(task_cfg=task_cfg)
```
