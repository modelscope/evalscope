# InfoVQA


## 概述

InfoVQA（信息图视觉问答）是一个用于评估 AI 模型基于信息密集型图像（如图表、图形、示意图、地图和信息图）回答问题能力的基准测试。该基准侧重于理解复杂的视觉信息呈现方式。

## 任务描述

- **任务类型**：信息图问答
- **输入**：信息图图像 + 自然语言问题
- **输出**：单个单词或短语形式的答案
- **领域**：数据可视化、信息图形、视觉推理

## 主要特点

- 聚焦于信息密集型视觉内容
- 涵盖图表、图形、示意图、地图和信息图
- 需要理解视觉布局和数据表示方式
- 测试信息提取与推理能力
- 问题复杂度从直接查找到推理不等

## 评估说明

- 默认使用 **验证集**（validation split）进行评估
- 主要指标：**ANLS**（Average Normalized Levenshtein Similarity，平均归一化编辑距离相似度）
- 答案格式应为 `"ANSWER: [ANSWER]"`
- 包含 OCR 文本提取结果作为元数据供分析使用
- 使用与 DocVQA 相同的数据源（InfographicVQA 子集）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `infovqa` |
| **数据集 ID** | [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `anls` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,801 |
| 提示词长度（平均） | 273.38 字符 |
| 提示词长度（最小/最大） | 222 / 390 字符 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 2,801 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 600x340 - 6250x9375 |
| 格式 | jpeg |


## 样例示例

**子集**：`InfographicVQA`

```json
{
  "input": [
    {
      "id": "03ab3147",
      "content": [
        {
          "text": "Answer the question according to the image using a single word or phrase.\nWhich social platform has heavy female audience?\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the question."
        },
        {
          "image": "[BASE64_IMAGE: png, ~249.7KB]"
        }
      ]
    }
  ],
  "target": "[\"pinterest\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "questionId": "98313",
    "answer_type": [
      "single span"
    ],
    "image_url": "https://blogs.constantcontact.com/wp-content/uploads/2019/03/Social-Media-Infographic.png",
    "ocr": "['{\"PAGE\": [{\"BlockType\": \"PAGE\", \"Geometry\": {\"BoundingBox\": {\"Width\": 0.9994840025901794, \"Height\": 0.9997748732566833, \"Left\": 0.0, \"Top\": 0.0}, \"Polygon\": [{\"X\": 0.0, \"Y\": 0.0}, {\"X\": 0.9994840025901794, \"Y\": 0.0}, {\"X\": 0.999484002590179 ... [TRUNCATED] ... 184143, \"Y\": 0.9778721332550049}, {\"X\": 0.5701684951782227, \"Y\": 0.9778721332550049}, {\"X\": 0.5701684951782227, \"Y\": 0.9896419048309326}, {\"X\": 0.47732439637184143, \"Y\": 0.9896419048309326}]}, \"Id\": \"43af6e92-c2ef-483c-b947-8b2d2073d756\"}]}']"
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
Answer the question according to the image using a single word or phrase.
{question}
The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the question.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets infovqa \
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
    datasets=['infovqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```