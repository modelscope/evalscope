# CrossSum-IN


## 概述

CrossSum-IN 评估对英语 BBC 新闻进行摘要的能力，输出语言为英语以及来自印度及周边地区的 29 种语言。

## 任务描述

- **任务类型**: 跨语言摘要
- **输入**: 一篇英语新闻文章
- **输出**: 使用子集对应语言生成的一句话摘要
- **领域**: BBC 新闻

## 主要特点

- 包含 12,463 篇测试文章，分布在 30 个摘要语言子集中。
- 默认使用论文中的提示词以及同一语言 `train` 划分中的一个示例。

## 评估说明

- 使用语料库 SacreBLEU chrF（字符顺序 6，单词顺序 0，beta 2）对第一个非空行进行评分。
- `indic` 类别的 `macro_score` 对每种语言赋予相等的权重。其 `score` 则根据样本数量进行加权。
- 文章未经截断处理。最长的示例需要较长的上下文窗口。
- 在 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可下用于非商业用途。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `crosssum_in` |
| **数据集ID** | [google/IndicGenBench_crosssum_in](https://modelscope.cn/datasets/google/IndicGenBench_crosssum_in/summary) |
| **论文** | [Paper](https://aclanthology.org/2024.acl-long.595/) |
| **标签** | `MultiLingual` |
| **指标** | `chrf` |
| **默认示例数** | 1-shot |
| **评估划分** | `test` |
| **训练划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 12,463 |
| 提示词长度 (平均值) | 8303.71 字符 |
| 提示词长度 (最小/最大值) | 1811 / 80198 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `en` | 500 | 8916.97 | 5066 | 76155 |
| `as` | 500 | 8917.97 | 5067 | 76156 |
| `awa` | 500 | 5879.97 | 2029 | 73118 |
| `bgc` | 500 | 8224.97 | 4374 | 75463 |
| `bho` | 500 | 8043.97 | 4193 | 75282 |
| `bn` | 155 | 6583.78 | 3341 | 21182 |
| `bo` | 499 | 6299.98 | 2445 | 73534 |
| `brx` | 500 | 5661.97 | 1811 | 72900 |
| `gbm` | 500 | 9939.97 | 6089 | 77178 |
| `gom` | 500 | 6356.97 | 2506 | 73595 |
| `gu` | 104 | 11224.04 | 7178 | 77591 |
| `hi` | 481 | 7252.13 | 3758 | 74847 |
| `hne` | 499 | 7089.83 | 3237 | 74326 |
| `hoj` | 500 | 9224.97 | 5374 | 76463 |
| `kn` | 500 | 8913.97 | 5063 | 76152 |
| `mai` | 499 | 9254.26 | 5404 | 76493 |
| `ml` | 500 | 8941.97 | 5091 | 76180 |
| `mni` | 497 | 6706.51 | 2855 | 73944 |
| `mr` | 118 | 9623.2 | 5447 | 76164 |
| `mup` | 500 | 10125.97 | 6275 | 77364 |
| `mwr` | 500 | 10087.97 | 6237 | 77326 |
| `ne` | 163 | 8686.5 | 5510 | 17316 |
| `or` | 500 | 8932.97 | 5082 | 76171 |
| `pa` | 136 | 10065.45 | 5956 | 27075 |
| `ps` | 154 | 8983.71 | 5350 | 76439 |
| `sa` | 499 | 6341.02 | 2488 | 73577 |
| `sat` | 500 | 9233.97 | 5383 | 76472 |
| `ta` | 315 | 6187.91 | 2738 | 73435 |
| `te` | 212 | 13614.55 | 9163 | 80198 |
| `ur` | 632 | 9770.29 | 6269 | 77358 |

## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "7d9c3fa5",
      "content": "I will first show a news article in English and then provide a summary of it in the English language.\n\nSummarize the following article: Dave LeeNorth America technology reporter Twitter said it removed 936 accounts it said were being used to  ... [TRUNCATED 11284 chars] ... y expensive. That's why, says Gagandeep Kang, a good vaccine for India should cost below 50 cents a shot, be plentifully available and delivered as a single dose. What do I need to know about the coronavirus? Follow Soutik on Twitter\nSummary:"
    }
  ],
  "target": "When it comes to vaccine making, India is a powerhouse.",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "article": "Soutik BiswasIndia correspondent It runs a massive immunisation programme, makes 60% of the world's vaccines and is home to half a dozen major manufacturers, including Serum Institute of India - the largest in the world. Not surprisingly, the ... [TRUNCATED 6503 chars] ... ld be very expensive. That's why, says Gagandeep Kang, a good vaccine for India should cost below 50 cents a shot, be plentifully available and delivered as a single dose. What do I need to know about the coronavirus? Follow Soutik on Twitter",
    "language": "en",
    "language_name": "English",
    "source_url": "https://www.bbc.com/news/world-asia-india-55048925",
    "target_url": "https://www.bbc.com/news/world-asia-india-55048925"
  }
}
```

*注意：部分内容为显示目的已截断。*

## 提示模板

**提示模板:**
```text
I will first show a news article in English and then provide a summary of it in the {language} language.

Summarize the following article: {article}
Summary:
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets crosssum_in \
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
    datasets=['crosssum_in'],
    dataset_args={
        'crosssum_in': {
            # subset_list: ['en', 'as', 'awa']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
