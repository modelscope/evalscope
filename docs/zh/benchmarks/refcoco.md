# RefCOCO

## 概述

RefCOCO 是一个用于训练和评估指代表达理解（Referring Expression Comprehension, REC）模型的数据集。它包含图像、目标物体的边界框，以及用自然语言自由形式描述 MSCOCO 图像中特定目标的表达。

## 任务描述

- **任务类型**：指代表达理解 / 图像描述生成
- **输入**：图像（含可视化）+ 指代表达
- **输出**：边界框坐标或描述文本
- **领域**：视觉定位、目标定位、图像理解

## 主要特性

- 通过 Amazon Mechanical Turk 标注创建
- 支持三种评估模式：
  - `bbox`：带边界框可视化的图像描述任务
  - `seg`：带分割掩码可视化的图像描述任务
  - `bbox_rec`：定位任务 —— 输出归一化的边界框坐标
- 表达式在复杂场景中唯一标识目标对象
- 包含多个子集：test、val、testA、testB

## 评估说明

- 通过 `eval_mode` 参数配置评估模式
- 提供多种指标进行综合评估：
  - 定位任务：IoU、ACC@0.1/0.3/0.5/0.7/0.9、Center_ACC
  - 描述生成任务：BLEU (1-4)、METEOR、ROUGE_L、CIDEr
- 边界框以归一化坐标输出：[x1/W, y1/H, x2/W, y2/H]
- 描述生成指标需依赖 pycocoevalcap 库

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `refcoco` |
| **数据集ID** | [lmms-lab/RefCOCO](https://modelscope.cn/datasets/lmms-lab/RefCOCO/summary) |
| **论文** | N/A |
| **标签** | `Grounding`, `ImageCaptioning`, `Knowledge`, `MultiModal` |
| **指标** | `IoU`, `ACC@0.1`, `ACC@0.3`, `ACC@0.5`, `ACC@0.7`, `ACC@0.9`, `Center_ACC`, `Bleu_1`, `Bleu_2`, `Bleu_3`, `Bleu_4`, `METEOR`, `ROUGE_L`, `CIDEr` |
| **默认示例数量** | 0-shot |
| **评估划分** | `N/A` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 17,596 |
| 提示词长度（平均） | 146 字符 |
| 提示词长度（最小/最大） | 146 / 146 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `test` | 5,000 | 146 | 146 | 146 |
| `val` | 8,811 | 146 | 146 | 146 |
| `testA` | 1,975 | 146 | 146 | 146 |
| `testB` | 1,810 | 146 | 146 | 146 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 13,785 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 300x176 - 640x640 |
| 格式 | jpeg |

## 样例示例

**子集**: `test`

```json
{
  "input": [
    {
      "id": "53a494fc",
      "content": [
        {
          "text": "Please carefully observe the area circled in the image and come up with a caption for the area.\nAnswer the question using a single word or phrase."
        },
        {
          "image": "[BASE64_IMAGE: jpeg, ~57.6KB]"
        }
      ]
    }
  ],
  "target": "['guy petting elephant', 'foremost person', 'green shirt']",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "469306",
    "iscrowd": 0,
    "file_name": "COCO_train2014_000000296747_0.jpg",
    "answer": [
      "guy petting elephant",
      "foremost person",
      "green shirt"
    ],
    "original_bbox": [
      59.04999923706055,
      93.23999786376953,
      375.0199890136719,
      362.5799865722656
    ],
    "bbox": [],
    "eval_mode": "bbox"
  }
}
```

## 提示模板

*未定义提示模板。*

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `eval_mode` | `str` | `bbox` | 控制 RefCOCO 使用的评估模式。bbox：图像描述任务，可视化原始图像与边界框；seg：图像描述任务，可视化原始图像与分割掩码；bbox_rec：定位任务，识别边界框坐标。可选值：['bbox', 'seg', 'bbox_rec'] |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets refcoco \
    --limit 10  # 正式评估时请移除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['refcoco'],
    dataset_args={
        'refcoco': {
            # subset_list: ['test', 'val', 'testA']  # 可选，评估指定子集
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```