# BoolQ-Indic


## 概述

BoolQ-Indic 是将 BoolQ 是/否阅读理解基准测试翻译为 10 种印度语言及英语的版本，用于评估多语言段落理解能力。

## 任务描述

- **任务类型**：多语言是/否阅读理解
- **输入**：一段文章 + 一个用 11 种语言之一提出的是/否问题
- **输出**：`Yes` 或 `No`
- **语言**：孟加拉语、英语、古吉拉特语、印地语、卡纳达语、马拉雅拉姆语、马拉地语、奥里亚语、旁遮普语、泰米尔语、泰卢固语

## 评估说明

- 默认配置使用 **0-shot** 评估（验证集）
- 使用 `subset_list` 来评估特定语言（例如 `['hi', 'ta']`），或使用 `limit` 限制样本数量 —— 默认完整运行包含全部 11 种语言共 35,970 个样本
- 设置 `few_shot_num` > 0 可启用少样本提示；示例从 `train` 划分中抽取
- 所有语言均包含在单一数据集配置中；此适配器根据 `language` 字段重新格式化数据

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `indic_boolq` |
| **数据集ID** | [sarvamai/boolq-indic](https://modelscope.cn/datasets/sarvamai/boolq-indic/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `MultiLingual`, `ReadingComprehension` |
| **指标** | `accuracy` |
| **默认样本数** | 0-shot |
| **评估划分** | `validation` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 35,970 |
| 提示词长度（平均） | 822.66 字符 |
| 提示词长度（最小/最大） | 275 / 5035 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `bn` | 3,270 | 801.95 | 294 | 2308 |
| `en` | 3,270 | 814.26 | 292 | 5035 |
| `gu` | 3,270 | 793.37 | 283 | 2105 |
| `hi` | 3,270 | 818.47 | 297 | 3078 |
| `kn` | 3,270 | 833.99 | 275 | 2920 |
| `ml` | 3,270 | 869.99 | 294 | 3558 |
| `mr` | 3,270 | 806.68 | 289 | 2593 |
| `or` | 3,270 | 787.68 | 306 | 1482 |
| `pa` | 3,270 | 804.93 | 295 | 1975 |
| `ta` | 3,270 | 904.01 | 297 | 3570 |
| `te` | 3,270 | 813.95 | 284 | 3312 |

## 样例示例

**子集**: `bn`

```json
{
  "input": [
    {
      "id": "0f04f0f7",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B.\n\nসকল জৈববস্তুই কমপক্ষে এই ধাপগুলোর মধ্য দিয়ে যায়: এগুলো চা ... [TRUNCATED 1099 chars] ...  বার্কলেতে  ছয়টি পৃথক গবেষণা বিশ্লেষণ করার পর, একটি গবেষণায় উপসংহারে আসা গেছে যে, ভুট্টা থেকে ইথানল উৎপাদনে পেট্রোলিয়ামের ব্যবহার গ্যাসোলিন উৎপাদনের তুলনায় অনেক কম।\n\nQuestion: ইথানল উৎপাদনের চেয়ে  তৈরিতে কি বেশি শক্তি লাগে??\n\nA) Yes\nB) No"
    }
  ],
  "choices": [
    "Yes",
    "No"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "bn",
  "metadata": {
    "language": "Bengali"
  }
}
```

*注：部分内容为显示目的已截断。*

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
    --datasets indic_boolq \
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
    datasets=['indic_boolq'],
    dataset_args={
        'indic_boolq': {
            # subset_list: ['bn', 'en', 'gu']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
