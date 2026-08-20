# BhashaBench-Multi (Krishi)


## 概述

BhashaBench-Multi (Krishi) 是一个领域特定的多项选择基准测试，用于评估大语言模型（LLM）在农业（Krishi）领域对22种印度语言的知识掌握情况。每个问题最初以英文编写，随后通过机器翻译（并附有基于LLM判断的翻译质量评分）转换为目标语言；本适配器使用的是翻译后的问题和选项。

## 任务描述

- **任务类型**：领域特定的多项选择问答
- **输入**：一道农业（Krishi）领域的四选一问题，使用22种印度语言之一
- **输出**：正确答案对应的字母
- **语言**：阿萨姆语、孟加拉语、博多语、多格拉语、古吉拉特语、印地语、卡纳达语、克什米尔语、孔卡尼语、迈蒂利语、马拉雅拉姆语、曼尼普尔语、马拉地语、尼泊尔语、奥里亚语、旁遮普语、梵语、桑塔利语、信德语、泰米尔语、泰卢固语、乌尔都语

## 主要特点

- 每种语言约14,963个问题，覆盖22种印度语言（每领域总计约33万）
- 从英文机器翻译而来，并附有基于LLM判断的翻译质量评分
- 覆盖印度22种官方语言，全部使用本地文字；不含英文子集
- 提供四个独立基准测试领域：阿育吠陀（Ayurveda）、金融（Finance）、农业（Krishi）、法律（Legal）

## 评估说明

- 默认配置采用 **0-shot** 评估（仅提供测试集）
- 可通过 `subset_list` 参数指定评估特定语言（例如 `['Hindi', 'Tamil']`），或使用 `limit` 限制样本数量 —— 每个领域在22种语言中每种语言约14,963个问题（总计约33万），因此完整评估所有语言将是一次大规模运行
- 本数据集不包含英文子集

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bhasha_bench_multi_krishi` |
| **数据集ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认Shots数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 338,910 |
| 提示词长度（平均） | 411.84 字符 |
| 提示词长度（最小/最大） | 207 / 2882 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Assamese` | 15,405 | 402.14 | 224 | 2265 |
| `Bengali` | 15,405 | 406.38 | 224 | 1506 |
| `Bodo` | 15,405 | 417.81 | 207 | 2186 |
| `Dogri` | 15,405 | 403.83 | 220 | 1988 |
| `Gujarati` | 15,405 | 397.6 | 224 | 1380 |
| `Hindi` | 15,405 | 407.8 | 224 | 1572 |
| `Kannada` | 15,405 | 407.21 | 224 | 1407 |
| `Kashmiri` | 15,405 | 442.73 | 245 | 2668 |
| `Konkani` | 15,405 | 402.57 | 222 | 1969 |
| `Maithili` | 15,405 | 393.7 | 224 | 1783 |
| `Malayalam` | 15,405 | 429.89 | 224 | 1661 |
| `Manipuri` | 15,405 | 436.33 | 240 | 2882 |
| `Marathi` | 15,405 | 406.47 | 224 | 1520 |
| `Nepali` | 15,405 | 402.05 | 224 | 1440 |
| `Oriya` | 15,405 | 392.2 | 221 | 1366 |
| `Punjabi` | 15,405 | 404.2 | 222 | 1536 |
| `Sanskrit` | 15,405 | 411.22 | 224 | 1412 |
| `Santhali` | 15,405 | 440.59 | 234 | 2773 |
| `Sindhi` | 15,405 | 392.36 | 224 | 1233 |
| `Tamil` | 15,405 | 441.53 | 224 | 2165 |
| `Telugu` | 15,405 | 412.75 | 224 | 1432 |
| `Urdu` | 15,405 | 409.12 | 224 | 2132 |

## 样例示例

**子集**: `Assamese`

```json
{
  "input": [
    {
      "id": "70df7242",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nইয়াকোনো বিশেষ স্থান আৰু সময়ত বায়ুমণ্ডলৰ অৱস্থা বা পৰিস্থিতি বুলি কোৱা হয়।\n\nA) জলবায়ু\nB) আবহাওয়া\nC) পৰ্যাৱৰণ\nD) বায়ুমণ্ডল"
    }
  ],
  "choices": [
    "জলবায়ু",
    "আবহাওয়া",
    "পৰ্যাৱৰণ",
    "বায়ুমণ্ডল"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "Assamese",
    "topic": null
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
    --datasets bhasha_bench_multi_krishi \
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
    datasets=['bhasha_bench_multi_krishi'],
    dataset_args={
        'bhasha_bench_multi_krishi': {
            # subset_list: ['Assamese', 'Bengali', 'Bodo']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
