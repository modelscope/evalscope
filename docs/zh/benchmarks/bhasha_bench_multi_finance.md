# BhashaBench-Multi (Finance)


## 概述

BhashaBench-Multi（Finance）是一个领域特定的多项选择题基准测试，用于评估大语言模型（LLM）在22种印度语言中对金融知识的掌握情况。每个问题最初以英文编写，随后通过机器翻译（并附有基于大语言模型判断的翻译质量评分）转换为目标语言；本适配器使用的是翻译后的问题和选项。

## 任务描述

- **任务类型**：领域特定的多项选择题问答
- **输入**：一道包含4个选项的金融问题，使用22种印度语言之一
- **输出**：正确答案对应的字母
- **语言**：阿萨姆语、孟加拉语、博多语、多格拉语、古吉拉特语、印地语、卡纳达语、克什米尔语、孔卡尼语、迈蒂利语、马拉雅拉姆语、曼尼普尔语、马拉地语、尼泊尔语、奥里亚语、旁遮普语、梵语、桑塔利语、信德语、泰米尔语、泰卢固日晚间乌尔都语

## 主要特点

- 每种语言约14,963道题目，覆盖22种印度语言（每个领域总计约33万题）
- 从英文机器翻译而来，并附有基于大语言模型判断的翻译质量评分
- 包含印度官方规定的22种语言，全部使用本地文字书写；不包含英文版本
- 提供四个独立的领域基准测试：阿育吠陀（Ayurveda）、金融（Finance）、农业（Krishi）、法律（Legal）

## 评估说明

- 默认配置采用 **0-shot** 评估（仅提供测试集）
- 可通过 `subset_list` 参数指定评估特定语言（例如 `['Hindi', 'Tamil']`），或使用 `limit` 限制样本数量——每个领域在22种语言中每种语言约有14,963道题目（总计约33万题），因此完整评估所有语言将是一次大规模运行
- 该数据集不包含英文版本

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `bhasha_bench_multi_finance` |
| **数据集ID** | [bharatgenai/BhashaBench-Multi](https://modelscope.cn/datasets/bharatgenai/BhashaBench-Multi/summary) |
| **论文** | 无 |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

*统计数据暂不可用。*

## 样例示例

*样例示例暂不可用。*

## 提示模板

**提示模板：**
```text
回答以下多项选择题。你的整个回复内容必须采用如下格式：'ANSWER: [LETTER]'（不含引号），其中 [LETTER] 是 {letters} 中的一个。

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
    --datasets bhasha_bench_multi_finance \
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
    datasets=['bhasha_bench_multi_finance'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
