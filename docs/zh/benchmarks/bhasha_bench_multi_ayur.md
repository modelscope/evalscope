# BhashaBench-Multi (Ayurveda)


## 概述

BhashaBench-Multi（Ayurveda）是一个领域特定的多项选择题基准测试，用于评估大语言模型（LLM）在22种印度语言中对阿育吠陀医学（Ayurvedic medicine）的知识掌握情况。每个问题最初以英文编写，随后通过机器翻译（并由LLM评估翻译质量得分）转换为目标语言；本适配器使用的是翻译后的题目和选项。

## 任务描述

- **任务类型**：领域特定的多项选择题问答
- **输入**：一道用22种印度语言之一编写的阿育吠陀医学问题，包含4个选项
- **输出**：正确答案对应的字母（A/B/C/D）
- **语言**：阿萨姆语、孟加拉语、博多语、多格拉语、古吉拉特语、印地语、卡纳达语、克什米尔语、孔卡尼语、迈蒂利语、马拉雅拉姆语、曼尼普尔语、马拉地语、尼泊尔语、奥里亚语、旁遮普语、梵语、桑塔利语、信德语、泰米尔语、泰卢固语、乌尔都语

## 主要特点

- 每种语言约14,963道题目，覆盖22种印度语言（每个领域总计约33万题）
- 题目从英文机器翻译而来，并附有LLM评估的翻译质量得分
- 包含印度官方规定的22种Scheduled Languages，全部使用本地文字书写；不包含英文版本
- 提供四个独立领域的基准测试：阿育吠陀（Ayurveda）、金融（Finance）、农业（Krishi）、法律（Legal）

## 评估说明

- 默认配置采用 **0-shot** 评估（仅提供测试集）
- 可通过 `subset_list` 参数指定评估特定语言（例如 `['Hindi', 'Tamil']`），或使用 `limit` 限制样本数量——每个领域在22种语言中每种语言约14,963题（总计约33万题），因此完整评估所有语言将是一次大规模运行
- 该数据集不包含英文版本

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bhasha_bench_multi_ayur` |
| **数据集ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认Shots数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 329,186 |
| 提示词长度（平均） | 317.8 字符 |
| 提示词长度（最小/最大） | 220 / 8370 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Assamese` | 14,963 | 325.76 | 229 | 4447 |
| `Bengali` | 14,963 | 313.28 | 231 | 1933 |
| `Bodo` | 14,963 | 315.55 | 222 | 1795 |
| `Dogri` | 14,963 | 308.39 | 225 | 1974 |
| `Gujarati` | 14,963 | 313.22 | 227 | 1526 |
| `Hindi` | 14,963 | 313.66 | 230 | 2018 |
| `Kannada` | 14,963 | 316.92 | 230 | 8305 |
| `Kashmiri` | 14,963 | 339.03 | 243 | 2102 |
| `Konkani` | 14,963 | 307.78 | 227 | 1819 |
| `Maithili` | 14,963 | 305.52 | 225 | 2142 |
| `Malayalam` | 14,963 | 330.73 | 236 | 1862 |
| `Manipuri` | 14,963 | 332.17 | 234 | 2247 |
| `Marathi` | 14,963 | 312.05 | 229 | 8370 |
| `Nepali` | 14,963 | 312.62 | 229 | 1825 |
| `Oriya` | 14,963 | 312.36 | 226 | 4092 |
| `Punjabi` | 14,963 | 309.98 | 220 | 926 |
| `Sanskrit` | 14,963 | 312.86 | 225 | 1232 |
| `Santhali` | 14,963 | 334.48 | 234 | 2283 |
| `Sindhi` | 14,963 | 309.44 | 226 | 852 |
| `Tamil` | 14,963 | 331.24 | 236 | 1031 |
| `Telugu` | 14,963 | 319.9 | 232 | 911 |
| `Urdu` | 14,963 | 314.71 | 227 | 921 |

## 样例示例

**子集**: `Assamese`

```json
{
  "input": [
    {
      "id": "4ac474ca",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nইমিউনজনিত বিকাৰসমূহৰ ভিতৰত আছে .....\n\nA) অতিরিক্ত সংবেদনশীলতা\nB) স্বয়ং-প্রতিরোধ ক্ষমতা জনিত ৰোগ\nC) রোগ প্রতিরোধ ক্ষমতাৰ অভাৱ\nD) এই সকলোবোৰ।"
    }
  ],
  "choices": [
    "অতিরিক্ত সংবেদনশীলতা",
    "স্বয়ং-প্রতিরোধ ক্ষমতা জনিত ৰোগ",
    "রোগ প্রতিরোধ ক্ষমতাৰ অভাৱ",
    "এই সকলোবোৰ।"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "Assamese",
    "topic": "Kayachikitsa"
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

### 使用CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bhasha_bench_multi_ayur \
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
    datasets=['bhasha_bench_multi_ayur'],
    dataset_args={
        'bhasha_bench_multi_ayur': {
            # subset_list: ['Assamese', 'Bengali', 'Bodo']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
