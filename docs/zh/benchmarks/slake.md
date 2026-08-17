# SLAKE


## 概述

SLAKE 是一个双语（英语/中文）的放射学视觉问答基准数据集，由医生基于 CT、MRI 和 X 光图像构建。问题既涵盖影像本身的纯视觉属性，也包含需要结合医学知识才能回答的内容。

## 任务描述

- **任务类型**：医学视觉问答（自由形式的简短回答）
- **输入**：一张放射学图像以及一条英语或中文问题
- **输出**：单个单词或短语，语言需与问题一致
- **领域**：放射学（胸部、腹部、脑部、骨盆、颈部）

## 主要特点

- 测试集包含 180 张图像上的 2,094 个问题，英语（1,061）和中文（1,033）大致均衡
- 每个问题均标注为 `OPEN`（自由回答）或 `CLOSED`（答案来自小型封闭集合，多为是/否），这是原始论文所采用的划分方式
- 问题涵盖十种语义类型：器官、位置、异常、知识图谱、模态、大小、平面、数量、颜色和形状
- 知识图谱类问题（`base_type=kvqa`）涉及病因、症状、治疗和功能等无法直接从图像中读取的信息

## 评估说明

- **主要指标**：**准确率（Accuracy）**，通过归一化后的精确匹配（exact match）与单一参考答案进行比较
- 结果按四个子集报告：`<language>_<open|closed>`，分为英语和中文两类；总体得分为样本加权平均值
- 归一化处理包括：转为小写、移除标点符号和括号内的附加说明、将 yes/no 的同义词统一为单一标签（因中文参考答案使用“是的 / 有 / 包含 / 可以”或“不是 / 没有 / 不包含 / 不可以”表达相同极性），并统一 X 光的不同拼写（包括中文的“X光 / X射线”），因为中文部分中的模态名称仍保留英文形式
- 答案从提示中要求的 `ANSWER:` 行读取；若模型未生成该行，则对整个回复进行归一化处理，因此仅重复问题的回复得分为 0
- 精确匹配设计上较为严格，与原始分类式评估一致：例如参考答案为 `Lung, Spinal Cord`、知识图谱中的治疗列表，或 `T2` 被回答为 `T2-weighted` 时，只有模型完全复现参考措辞才算正确，因此知识图谱类开放问题的准确率预期较低
- 图像以单个 `imgs.zip` 文件（约 200 MB）提供，并直接从压缩包中读取
- [论文](https://arxiv.org/abs/2102.09542) | [项目主页](https://www.med-vqa.com/slake/)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `slake` |
| **数据集ID** | [evalscope/SLAKE](https://modelscope.cn/datasets/evalscope/SLAKE/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2102.09542) |
| **标签** | `Medical`, `MultiModal`, `QA` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,094 |
| 提示词长度（平均） | 130.2 字符 |
| 提示词长度（最小/最大） | 60 / 257 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `en_open` | 645 | 195.09 | 168 | 257 |
| `en_closed` | 416 | 187.99 | 162 | 253 |
| `zh_open` | 613 | 67.07 | 61 | 79 |
| `zh_closed` | 420 | 65.44 | 60 | 82 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 总图像数 | 2,094 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 240x240 - 1024x1024 |
| 格式 | jpeg |


## 样例示例

**子集**: `en_open`

```json
{
  "input": [
    {
      "id": "411b63eb",
      "content": [
        {
          "image": "[BASE64_IMAGE: jpeg, ~63.2KB]"
        },
        {
          "text": "What modality is used to take this image?\nAnswer the question with a single word or phrase in English.\nThe last line of your response must be of the form \"ANSWER: <answer>\" (without quotes)."
        }
      ]
    }
  ],
  "target": "CT",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "qid": 11934,
    "img_name": "xmlab102/source.jpg",
    "answer_type": "OPEN",
    "content_type": "Modality",
    "modality": "CT"
  }
}
```

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets slake \
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
    datasets=['slake'],
    dataset_args={
        'slake': {
            # subset_list: ['en_open', 'en_closed', 'zh_open']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
