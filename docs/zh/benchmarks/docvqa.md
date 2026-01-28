# DocVQA


## 概述

DocVQA（Document Visual Question Answering，文档视觉问答）是一个用于评估 AI 系统根据文档图像（如扫描页面、表单、发票和报告）回答问题能力的基准测试。该任务不仅要求简单的文本提取，还需要理解复杂的文档布局、结构和视觉元素。

## 任务描述

- **任务类型**：文档视觉问答
- **输入**：文档图像 + 自然语言问题
- **输出**：从文档中提取的单个词或短语作为答案
- **领域**：文档理解、OCR、版面理解

## 主要特点

- 涵盖多种文档类型（表单、发票、信函、报告）
- 需要理解文档的版面与结构
- 同时考察文本提取能力和上下文推理能力
- 问题要求定位并解读特定信息
- 结合了 OCR 能力与视觉理解能力

## 评估说明

- 默认使用 **验证集**（validation split）进行评估
- 主要指标：**ANLS**（Average Normalized Levenshtein Similarity，平均归一化编辑距离相似度）
- 答案格式应为 `"ANSWER: [ANSWER]"`
- ANLS 指标可容忍轻微的 OCR 或拼写差异
- 每个问题可能接受多个有效答案

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `docvqa` |
| **数据集 ID** | [lmms-lab/DocVQA](https://modelscope.cn/datasets/lmms-lab/DocVQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MultiModal`, `QA` |
| **指标** | `anls` |
| **默认示例数量** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,349 |
| 提示词长度（平均） | 254.82 字符 |
| 提示词长度（最小/最大） | 220 / 354 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 5,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 593x294 - 5367x7184 |
| 格式 | png |


## 样例示例

**子集**: `DocVQA`

```json
{
  "input": [
    {
      "id": "002390bd",
      "content": [
        {
          "text": "Answer the question according to the image using a single word or phrase.\nWhat is the ‘actual’ value per 1000, during the year 1975?\nThe last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the question."
        },
        {
          "image": "[BASE64_IMAGE: png, ~1.2MB]"
        }
      ]
    }
  ],
  "target": "[\"0.28\"]",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "questionId": "49153",
    "question_types": [
      "figure/diagram"
    ],
    "docId": 14465,
    "ucsf_document_id": "pybv0228",
    "ucsf_document_page_no": "81"
  }
}
```

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
    --datasets docvqa \
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
    datasets=['docvqa'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```