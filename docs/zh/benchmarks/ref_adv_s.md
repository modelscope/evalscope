# Ref-Adv-s


## 概述

Ref-Adv-s 是 Ref-Adv 基准测试中公开的 1,142 个样例子集。该基准测试旨在评估多模态大语言模型是否能够从同类视觉干扰物中准确识别目标对象，而非依赖简单的视觉定位捷径。

## 任务描述

- **任务类型**：指代表达理解 / 视觉定位
- **输入**：一张图像和一个英文指代表达式
- **输出**：一个或多个 JSON 格式的边界框，其中第一个边界框用于评分
- **领域**：包含同类干扰物的 COCO 和 OpenImages 场景

## 主要特点

- 包含从 5,000 个样例的 Ref-Adv 基准测试中采样的 1,142 个公开样例
- 包含人工撰写和模型辅助生成的表达式、显式否定词，且每个样例至少包含两个干扰物
- 保留官方的 `direct` 和思维链（`cot`）提示模式
- 使用数据集唯一的 `train` 划分作为评估划分

## 评估说明

- 报告官方指标 `Acc@0.5`、`Acc@0.75` 和 `Acc@0.9`，基于第一个解析出的边界框与真实框的 IoU 计算
- 同时报告干扰物数量分组（`2-3`、`4-6` 和 `>=7`）下的 `Acc@0.5` 指标
- 使用官方指定的键搜索顺序，解析响应中最后一个有效的围栏式 JSON 对象，或以非围栏式 JSON 值结尾的内容
- 若首次解析失败，将触发官方的一轮格式修复提示；若第二次仍失败，则该样本得分为零
- 对于 Qwen2.5-VL，请将 `pred_box_format` 设置为 `abs_xyxy`；对于 Qwen3-VL/Qwen3.5，请设置为 `norm_1000_xyxy`；官方评估器也支持 `norm_1_xyxy`
- [论文](https://arxiv.org/abs/2602.23898) | [GitHub](https://github.com/dddraxxx/Ref-Adv)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ref_adv_s` |
| **数据集ID** | [evalscope/ref-adv-s](https://modelscope.cn/datasets/evalscope/ref-adv-s/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2602.23898) |
| **标签** | `Grounding`, `MultiModal`, `Reasoning` |
| **指标** | `ACC@0.5`, `ACC@0.75`, `ACC@0.9`, `2-3/ACC@0.5`, `4-6/ACC@0.5`, `>=7/ACC@0.5` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,142 |
| 提示词长度（平均） | 177.67 字符 |
| 提示词长度（最小/最大） | 136 / 282 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 1,142 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 240x320 - 1024x1024 |
| 格式 | jpeg |


## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "d3a7e578",
      "content": [
        {
          "text": "<image>\nLocate every object that matches the description \"the computer screen that is in the middle vertically of the three stacks\" in the image. Report bbox coordinates in JSON format."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~118.1KB]"
        }
      ]
    }
  ],
  "target": "[297.0, 345.0, 427.0, 440.0]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "row_idx": 0,
    "file_name": "000000547144.jpg",
    "image_source": "coco_val2017",
    "human_authored": true,
    "use_negation": false,
    "distractor_count": 5,
    "target_box_normalized": [
      0.4640625,
      0.71875,
      0.6671875,
      0.9166666666666666
    ],
    "sent_size": [
      640,
      480
    ],
    "retry_followup_used": false
  }
}
```

## 提示模板

**提示模板：**
```text
<image>
Locate every object that matches the description "{ref_sentence}" in the image. Report bbox coordinates in JSON format.
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `prompt_mode` | `str` | `direct` | 官方提示模式。选项：['direct', 'cot'] |
| `pred_box_format` | `str` | `norm_1000_xyxy` | 被评估模型输出的坐标格式。选项：['abs_xyxy', 'norm_1000_xyxy', 'norm_1_xyxy'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ref_adv_s \
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
    datasets=['ref_adv_s'],
    dataset_args={
        'ref_adv_s': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
