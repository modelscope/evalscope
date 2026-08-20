# BhashaBench-Multi (Legal)


## 概述

BhashaBench-Multi (Legal) 是一个领域特定的多项选择题基准测试，用于评估大语言模型（LLM）在22种印度语言中对印度法律知识的掌握情况。每个问题最初以英文编写，随后通过机器翻译（并附有基于LLM判断的翻译质量评分）转换为目标语言；本适配器使用的是翻译后的问题和选项。

## 任务描述

- **任务类型**：领域特定的多项选择题问答
- **输入**：一道印度法律问题，包含4个选项，使用22种印度语言之一
- **输出**：正确答案对应的字母
- **语言**：阿萨姆语、孟加拉语、博多语、多格拉语、古吉拉特语、印地语、卡纳达语、克什米尔语、孔卡尼语、迈蒂利语、马拉雅拉姆语、曼尼普尔语、马拉地语、尼泊尔语、奥里亚语、旁遮普语、梵语、桑塔利语、信德语、泰米尔语、泰卢固语、乌尔都语

## 主要特点

- 每种语言约14,963道题目，覆盖22种印度语言（每个领域总计约33万题）
- 从英文机器翻译而来，并附带基于LLM判断的翻译质量评分
- 覆盖印度宪法规定的22种预定语言，全部使用本地文字书写；不包含英文版本
- 提供四个独立领域的基准测试：阿育吠陀（Ayurveda）、金融（Finance）、农业（Krishi）、法律（Legal）

## 评估说明

- 默认配置采用 **0-shot** 评估（仅提供测试集）
- 可通过 `subset_list` 参数指定评估特定语言（例如 `['Hindi', 'Tamil']`），或使用 `limit` 限制样本数量 —— 每个领域在22种语言中每种语言约14,963道题（总计约33万题），因此完整评估所有语言将是一次大规模运行
- 该数据集不包含英文版本

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bhasha_bench_multi_legal` |
| **数据集ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认Shots数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 536,030 |
| 提示词长度（平均） | 490.89 字符 |
| 提示词长度（最小/最大） | 225 / 6384 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Assamese` | 24,365 | 475.08 | 232 | 2556 |
| `Bengali` | 24,365 | 482.5 | 235 | 2066 |
| `Bodo` | 24,365 | 521.23 | 225 | 4608 |
| `Dogri` | 24,365 | 487.72 | 225 | 4432 |
| `Gujarati` | 24,365 | 463.64 | 232 | 1954 |
| `Hindi` | 24,365 | 489.37 | 232 | 2202 |
| `Kannada` | 24,365 | 475.35 | 232 | 2068 |
| `Kashmiri` | 24,365 | 514.54 | 242 | 5037 |
| `Konkani` | 24,365 | 473.95 | 225 | 4000 |
| `Maithili` | 24,365 | 473.11 | 225 | 4011 |
| `Malayalam` | 24,365 | 511.29 | 236 | 2218 |
| `Manipuri` | 24,365 | 548.36 | 238 | 6384 |
| `Marathi` | 24,365 | 487.86 | 232 | 2113 |
| `Nepali` | 24,365 | 475.87 | 234 | 2058 |
| `Oriya` | 24,365 | 458.86 | 232 | 1936 |
| `Punjabi` | 24,365 | 483.03 | 232 | 2138 |
| `Sanskrit` | 24,365 | 489.0 | 233 | 1979 |
| `Santhali` | 24,365 | 549.75 | 233 | 5074 |
| `Sindhi` | 24,365 | 455.74 | 234 | 1830 |
| `Tamil` | 24,365 | 522.97 | 237 | 2479 |
| `Telugu` | 24,365 | 481.32 | 234 | 1992 |
| `Urdu` | 24,365 | 479.13 | 235 | 2120 |

## 样例示例

**子集**: `Assamese`

```json
{
  "input": [
    {
      "id": "e631cc6e",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nকোনো আদেশ প্ৰকাশ কৰাৰ পূৰ্বতে কোনো সমস্যা সংশোধন কৰাৰ বা নতুন সমস্যা উত্থাপন কৰাৰ ক্ষমতা আদালতৰ ওচৰত থাকে, আৰু এই ক্ষমতা দিয়া হয় দেৱানী প্রক্রিয়া বিধি, ১৯০৮-ৰ কোনটো ব্যৱস্থাৰ দ্বাৰা?\n\nA) অধ্যায় ১৪, বিধি ১\nB) অধ্যায় ১৪, বিধি ৫\nC) অধ্যায় XIV, বিধি ৬\nD) ধাৰা ১৫১"
    }
  ],
  "choices": [
    "অধ্যায় ১৪, বিধি ১",
    "অধ্যায় ১৪, বিধি ৫",
    "অধ্যায় XIV, বিধি ৬",
    "ধাৰা ১৫১"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "Assamese",
    "topic": "Procedural Law"
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

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bhasha_bench_multi_legal \
    --limit 10  # 正式评估时请删除此行
```

### 使用Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['bhasha_bench_multi_legal'],
    dataset_args={
        'bhasha_bench_multi_legal': {
            # subset_list: ['Assamese', 'Bengali', 'Bodo']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
