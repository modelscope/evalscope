# ARC-Challenge-Indic


## 概述

ARC-Challenge-Indic 是将 AI2 推理挑战赛（ARC-Challenge）科学问答基准翻译成 10 种印度语言，并保留原始英文版本，共涵盖 11 种语言，用于评估多语言科学推理能力。

## 任务描述

- **任务类型**：多语言多项选择科学问答
- **输入**：以 11 种语言之一呈现的科学问题及其选项
- **输出**：正确答案对应的字母
- **语言**：孟加拉语、英语、古吉拉特语、印地语、卡纳达语、马拉雅拉姆语、马拉地语、奥里亚语、旁遮普语、泰米尔语、泰卢固语

## 评估说明

- 默认配置使用 **0-shot** 评估（测试集）
- 使用 `subset_list` 参数可指定评估特定语言（例如 `['hi', 'ta']`）
- 所用题目与 `arc` 基准（Challenge 分割）相同，每种语言均经过机器或人工翻译

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `arc_indic` |
| **数据集ID** | [sarvamai/arc-challenge-indic](https://modelscope.cn/datasets/sarvamai/arc-challenge-indic/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `MultiLingual`, `Reasoning` |
| **指标** | `accuracy` |
| **默认样本数** | 0-shot |
| **评估分割** | `test` |
| **训练分割** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 12,647 |
| 提示词长度（平均） | 448.01 字符 |
| 提示词长度（最小/最大） | 236 / 2053 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `bn` | 1,150 | 432.51 | 242 | 1137 |
| `en` | 1,147 | 454.88 | 253 | 1111 |
| `gu` | 1,150 | 426.57 | 243 | 1098 |
| `hi` | 1,150 | 443.47 | 236 | 1162 |
| `kn` | 1,150 | 456.08 | 245 | 1199 |
| `ml` | 1,150 | 473.31 | 239 | 2053 |
| `mr` | 1,150 | 434.22 | 242 | 1133 |
| `or` | 1,150 | 440.04 | 243 | 1374 |
| `pa` | 1,150 | 443.35 | 236 | 1132 |
| `ta` | 1,150 | 479.12 | 243 | 1295 |
| `te` | 1,150 | 444.53 | 244 | 1172 |

## 样例示例

**子集**: `bn`

```json
{
  "input": [
    {
      "id": "f750462b",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nএকজন খগোলবিদ পর্যবেক্ষণ করেন যে একটি উল্কা পতনের পরে একটি গ্রহের ঘূর্ণন গতি বেড়ে যায়। ঘূর্ণন বৃদ্ধির ফলে কোন প্রভাবটি সবচেয়ে বেশি সম্ভাব্য?\n\nA) গ্রহের ঘনত্ব কমে যাবে।\nB) গ্রহীয় বছরগুলি আরও দীর্ঘ হবে।\nC) গ্রহের দিনগুলি ছোট হয়ে যাবে।\nD) গ্রহের মাধ্যাকর্ষণ শক্তি আরও বৃদ্ধি পাবে।"
    }
  ],
  "choices": [
    "গ্রহের ঘনত্ব কমে যাবে।",
    "গ্রহীয় বছরগুলি আরও দীর্ঘ হবে।",
    "গ্রহের দিনগুলি ছোট হয়ে যাবে।",
    "গ্রহের মাধ্যাকর্ষণ শক্তি আরও বৃদ্ধি পাবে।"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": "Mercury_7175875",
    "language": "Bengali"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets arc_indic \
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
    datasets=['arc_indic'],
    dataset_args={
        'arc_indic': {
            # subset_list: ['bn', 'en', 'gu']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
