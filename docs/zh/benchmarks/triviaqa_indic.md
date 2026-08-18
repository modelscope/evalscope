# TriviaQA-Indic-MCQ


## 概述

TriviaQA-Indic-MCQ 将 TriviaQA 的常识问答题重新格式化为四选一的多项选择题，并翻译成 10 种印度语言及英语，用于评估模型在多语言环境下的世界知识回忆能力。

## 任务描述

- **任务类型**：多语言多项选择常识问答
- **输入**：以 11 种语言之一呈现的常识问题及其 4 个选项
- **输出**：正确答案对应的字母
- **语言**：孟加拉语、英语、古吉拉特语、印地语、卡纳达语、马拉雅拉姆语、马拉地语、奥里亚语、旁遮普语、泰米尔语、泰卢固语

## 评估说明

- 默认配置使用 **0-shot** 评估（仅提供验证集）
- 可通过 `subset_list` 参数指定评估特定语言（例如 `['hi', 'ta']`），或使用 `limit` 参数限制样本数量 —— 完整默认运行包含全部 11 种语言，每种语言约 1.8 万个样本（总计约 19.8 万）

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `triviaqa_indic` |
| **数据集ID** | [sarvamai/trivia-qa-indic-mcq](https://modelscope.cn/datasets/sarvamai/trivia-qa-indic-mcq/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 197,384 |
| 提示词长度（平均） | 353.95 字符 |
| 提示词长度（最小/最大） | 247 / 1267 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `bn` | 17,944 | 351.81 | 253 | 1131 |
| `en` | 17,944 | 347.69 | 256 | 1158 |
| `gu` | 17,944 | 347.73 | 247 | 1117 |
| `hi` | 17,944 | 349.05 | 258 | 1141 |
| `kn` | 17,944 | 359.89 | 253 | 1164 |
| `ml` | 17,944 | 366.11 | 257 | 1267 |
| `mr` | 17,944 | 350.9 | 253 | 1078 |
| `or` | 17,944 | 349.44 | 255 | 941 |
| `pa` | 17,944 | 345.48 | 253 | 1157 |
| `ta` | 17,944 | 368.55 | 248 | 1198 |
| `te` | 17,944 | 356.75 | 254 | 1259 |

## 样例示例

**子集**: `bn`

```json
{
  "input": [
    {
      "id": "eff7630c",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nচিপমঙ্কসের পিছনে লোকটি কে ছিল?\n\nA) ডেভিড সেভিল\nB) জাগরেব শহর - ক্রোয়েশিয়া প্রজাতন্ত্র\nC) পবিত্র ক্রুসেড\nD) উপাদান (অ্যালবাম)"
    }
  ],
  "choices": [
    "ডেভিড সেভিল",
    "জাগরেব শহর - ক্রোয়েশিয়া প্রজাতন্ত্র",
    "পবিত্র ক্রুসেড",
    "উপাদান (অ্যালবাম)"
  ],
  "target": "A",
  "id": 0,
  "group_id": 0,
  "metadata": {
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
    --datasets triviaqa_indic \
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
    datasets=['triviaqa_indic'],
    dataset_args={
        'triviaqa_indic': {
            # subset_list: ['bn', 'en', 'gu']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
