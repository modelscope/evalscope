# Maritime-OCR-Bench


## 概述

Maritime-OCR-Bench 是一个全面的评估基准，用于衡量多模态大模型在 OCR 相关任务上的能力。当前发布的数据集包含 1,888 个经过人工精心整理的样本，涵盖五种任务类型。

## 任务类型

- **VQA**：针对文档/场景图像的视觉问答
- **IE**：要求严格 JSON 格式输出的信息抽取
- **parsing**：从图像中进行文本识别与解析
- **json1**：采用 JSON v1 结构化格式的文本检测
- **json2**：采用 JSON v2 结构化格式的文本检测

## 评估指标

每种任务类型采用专门的评分方法：
- VQA/parsing：多维文本相似度（编辑距离、字符级 F1、最长公共子序列 F1、表格感知相似度）
- IE：文本覆盖率 + JSON 严格性（0.5 * coverage + 0.5 * json_strict）
- json1/json2：DIoU 布局得分 + 文本得分（0.7 * diou + 0.3 * text）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `maritime_ocr_bench` |
| **数据集ID** | [HiDolphin/MaritimeOCRBench](https://modelscope.cn/datasets/HiDolphin/MaritimeOCRBench/summary) |
| **论文** | N/A |
| **标签** | `MultiModal`, `QA` |
| **指标** | `score` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,888 |
| 提示词长度（平均） | 102.91 字符 |
| 提示词长度（最小/最大） | 23 / 288 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `IE` | 471 | 23 | 23 | 23 |
| `VQA` | 471 | 39.65 | 28 | 141 |
| `parsing` | 472 | 80 | 80 | 80 |
| `json1` | 237 | 256.1 | 248 | 288 |
| `json2` | 237 | 279.9 | 248 | 288 |

**图像统计数据：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 1,888 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 108x50 - 4030x4075 |
| 格式 | jpeg, png |

## 样例示例

**子集**: `IE`

```json
{
  "input": [
    {
      "id": "1d6ce119",
      "content": [
        {
          "text": "请提取所有关键信息，并以 JSON 格式返回。"
        },
        {
          "image": "[BASE64_IMAGE: png, ~550.9KB]"
        }
      ]
    }
  ],
  "target": "{\n  \"Document ID\": \"WiCE Error Message Description\",\n  \"Revision\": \"REV 1\",\n  \"Date\": \"2024-09-12\",\n  \"Company_Logo_Text\":\"WIN GD\",\n  \"Error Messages\": [\n    {\n      \"ID Number\": \"COFD-52\",\n      \"Designation\": \"Fuel Pump Control Signal #2 Fa ... [TRUNCATED 1668 chars] ... nt must be within 4 ~ 20mA.\\n• If necessary, the sensor can be replaced with new one (Caution: before dismantling, do depressurize rail)\"\n    }\n  ],\n  \"Footer\": \"T_PC-Drawing_Portrait | Release: 3.10 (2024-05-15)\",\n  \"Page\": \"Page 20 of 83\"\n}",
  "id": 0,
  "group_id": 0,
  "subset_key": "IE",
  "metadata": {
    "task_type": "IE",
    "prompt": "<image>请提取所有关键信息，并以 JSON 格式返回。",
    "images": [
      "images/580968569085497344_f33f02a81a.png"
    ]
  }
}
```

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets maritime_ocr_bench \
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
    datasets=['maritime_ocr_bench'],
    dataset_args={
        'maritime_ocr_bench': {
            # subset_list: ['IE', 'VQA', 'parsing']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```