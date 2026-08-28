# MedXpertQA


## 概述

MedXpertQA 是一个专家级医学多项选择基准测试，旨在评估高级医学知识与推理能力。该基准包含独立的纯文本（Text-only）和多模态（Multimodal, MM）两个赛道，题目源自具有挑战性的医学考试试题，并经由持证医师审核。

## 任务描述

- **任务类型**：单答案医学多项选择题
- **输入**：一道临床或生物医学问题及其选项，可选附带最多六张图像
- **输出**：一个答案字母（Text 赛道为 A-J，MM 赛道为 A-E）
- **领域**：涵盖17个医学专科和11个人体系统

## 主要特点

- 测试集包含4,450道题目：其中2,450道为Text题目（含十个选项），2,000道为MM题目（含五个选项）
- MM赛道包含放射影像、病理切片、光学图像、照片、示意图、图表、表格、文档及生命体征图像
- 所有题目均标注了医学任务类型、人体系统和问题类型；其中3,307道测试题侧重推理能力，1,143道侧重理解能力
- 题目经过难度筛选、选项增强、数据泄露缓解以及多轮专家评审

## 评估说明

- 主要指标：**准确率（Accuracy）**，通过预测答案字母与标准答案的精确匹配计算
- 默认提示词采用 EvalScope 的零样本思维链（zero-shot chain-of-thought）模板，保留官方指定的逐步推理指令及严格的答案字母评分格式
- 应将 `max_tokens` 设置得足够高，以确保模型能完整输出所需的最终行 `ANSWER: [LETTER]`；否则，若推理过程被截断，解析器可能回退到提取最后一个有效的大写字母作为答案
- 结果分别报告 Text 和 MM 子集的表现，并通过样本加权聚合计算整体得分
- MM图像存储在 `images.zip`（约517 MB）中，直接从压缩包读取，无需额外解压副本
- 公开数据集共包含4,460条记录（含10个开发样例）；本集成仅评估其中4,450道预留的测试题
- [论文](https://arxiv.org/abs/2501.18362) | [GitHub](https://github.com/TsinghuaC3I/MedXpertQA)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `medxpertqa` |
| **数据集ID** | [evalscope/MedXpertQA](https://modelscope.cn/datasets/evalscope/MedXpertQA/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2501.18362) |
| **标签** | `MCQ`, `Medical`, `MultiModal`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 4,450 |
| 提示词长度（平均） | 1135.22 字符 |
| 提示词长度（最小/最大） | 346 / 4771 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Text` | 2,450 | 1337.92 | 435 | 4771 |
| `MM` | 2,000 | 886.91 | 346 | 2335 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 2,852 |
| 每样本图像数 | 最小: 1, 最大: 6, 平均: 1.43 |
| 分辨率范围 | 323x34 - 4248x2144 |
| 格式 | jpeg, png |


## 样例示例

**子集**: `Text`

```json
{
  "input": [
    {
      "id": "3f1d2f2a",
      "content": "You are a helpful medical assistant."
    },
    {
      "id": "1a9f9143",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J. Think step by step before answering.\n\nWhich pat ... [TRUNCATED 885 chars] ... ere posterior wear undergoing shoulder arthroplasty\nI) 58-year-old male with glenoid retroversion of 12-degrees undergoing shoulder arthroplasty\nJ) 55-year-old male with glenoid retroversion of 8-degrees undergoing total shoulder arthroplasty"
        }
      ]
    }
  ],
  "choices": [
    "70-year-old male with glenoid retroversion of 18-degrees undergoing shoulder arthroplasty",
    "70-year-old female with humeral anteversion of 13-degrees undergoing shoulder arthroplasty",
    "63-year-old female with glenoid retroversion of 22-degrees and mild posterior wear undergoing shoulder arthroplasty",
    "65-year-old female with glenoid retroversion of 25-degrees undergoing shoulder arthroplasty",
    "65-year-old female with a glenoid retroversion of 13-degrees undergoing shoulder arthroplasty",
    "68-year-old female with glenoid retroversion of 20-degrees undergoing reverse shoulder arthroplasty",
    "72-year-old male with glenoid retroversion of 15-degrees undergoing shoulder arthroplasty",
    "65-year-old female with glenoid retroversion of 30-degrees and severe posterior wear undergoing shoulder arthroplasty",
    "58-year-old male with glenoid retroversion of 12-degrees undergoing shoulder arthroplasty",
    "55-year-old male with glenoid retroversion of 8-degrees undergoing total shoulder arthroplasty"
  ],
  "target": "E",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "Text-0",
    "medical_task": "Basic Science",
    "body_system": "Skeletal",
    "question_type": "Reasoning",
    "images": []
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**系统提示：**
```text
You are a helpful medical assistant.
```

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets medxpertqa \
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
    datasets=['medxpertqa'],
    dataset_args={
        'medxpertqa': {
            # subset_list: ['Text', 'MM']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
