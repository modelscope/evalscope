# CrossSum-IN

## 概述

CrossSum-IN 是 IndicGenBench（Google Research，ACL 2024）中的跨语言摘要任务。它基于 CrossSum 构建，将英文 BBC 新闻文章与人工撰写的 29 种印度语言摘要（其中许多是低资源语言）以及英文摘要配对。

## 任务描述

- **任务类型**：跨语言摘要
- **输入**：一篇英文新闻文章
- **输出**：用该子集语言写成的一句话摘要
- **语言**：阿萨姆语、阿瓦德语、孟加拉语、博杰普尔语、博多语、恰蒂斯加尔语、加瓦尔语、古吉拉特语、哈里亚纳语、印地语、卡纳达语、孔卡尼语、迈蒂利语、马尔维语、马拉雅拉姆语、曼尼普尔语、马拉地语、马尔瓦里语、尼泊尔语、奥里亚语、普什图语、旁遮普语、拉贾斯坦语、梵语、桑塔利语、泰米尔语、泰卢固语、藏语和乌尔都语，以及英语

## 主要特点

- 30 个子集，按摘要语言命名（`en`、`hi`、`sat` 等），每个子集在其发布的 `test` 文件上评估（每种语言 104 到 632 篇文章，共计 12,463 篇）
- 同一语言的 `train` 文件（100 篇文章，`gbm` 为 99 篇）提供上下文示例
- 结果分为 `indic`（29 种印度语言）和 `english` 两组，论文中二者分开报告

## 评估说明

- 默认配置使用 **1-shot** 评估，即论文的主要设置。示例为同一语言的第一篇 `train` 文章，因此每个模型看到的提示完全相同；设置 `few_shot_num=0` 可进行零样本评估
- 提示（论文表 15），语言名称采用论文表 7 中的英文名称：
  `I will first show a news article in English and then provide a summary of it in the {language} language.\n\n`
  `Summarize the following article: {article}\nSummary:`。示例使用相同格式，后接 ` {summary}`，再空一行，然后是测试文章
- 与论文一致，文章不做截断：平均约 4,200 个字符，5% 超过 8,200 个字符，另有 28 篇（0.2%）超过 32,768 个字符（约 8k token），上下文窗口较短的模型无法处理这些文章
- 只对输出的第一个非空行进行评分；与论文一致，请使用贪心解码（`temperature=0`）
- 指标：SacreBLEU 默认设置下的 **chrF**（字符阶数 6，词阶数 0，beta 为 2），按每种语言在语料库级别计算。印度语言得分是 29 种印度语言的宏平均，报告为 `indic` 类别的 `macro_score`（各语言样本数不同，因此按样本数加权的类别 `score` 是另一个数值）
- 每个样本还会记录其句子级 chrF 以便检查；报告中只给出语料库级得分
- 许可证：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，继承自 CrossSum：仅限非商业用途

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `crosssum_in` |
| **数据集ID** | [google/IndicGenBench_crosssum_in](https://modelscope.cn/datasets/google/IndicGenBench_crosssum_in/summary) |
| **论文** | [论文](https://aclanthology.org/2024.acl-long.595/) |
| **标签** | `MultiLingual` |
| **指标** | `chrf` |
| **默认样本数** | 1-shot |
| **评估分割** | `test` |
| **训练分割** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 12,463 |
| 提示词长度（平均） | 8303.71 字符 |
| 提示词长度（最小/最大） | 1811 / 80198 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
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
      "id": "4a179703",
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

*注：部分内容已截断显示。*

## 提示模板

**提示模板：**
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
