# MMLU-Indic


## 概述

MMLU-Indic 是 Sarvam AI 将 MMLU（Massive Multitask Language Understanding，大规模多任务语言理解）基准翻译成
10 种印度语言的版本，每种语言都有原生文字和罗马化（拉丁字母转写）两个变体，另外还包含原始英文版本。它是 Sarvam
翻译的第四个学术基准，其余三个为 `arc_indic`、`triviaqa_indic` 和 `indic_boolq`。

## 任务描述

- **任务类型**：多语言多项选择问答
- **输入**：21 种语言/文字变体之一的问题及选项
- **输出**：正确答案的字母（A-D）
- **领域**：MMLU 的 57 个学科（STEM、人文、社会科学及其他）

## 主要特点

- 21 个子集：原生文字的英语、孟加拉语、古吉拉特语、印地语、卡纳达语、马拉雅拉姆语、马拉地语、奥里亚语、旁遮普语、
  泰米尔语和泰卢固语，以及以 `<lang>_roman` 命名的 10 种印度语言罗马化版本
- 原生文字子集包含完整的 MMLU 测试集（14,042 道题），并以其 285 道题的 dev 集作为 `validation` 划分；
  罗马化子集包含 13,096-14,005 道测试题，没有 `validation` 划分
- 数据集未保留 MMLU 的学科标签，因此结果仅按语言报告
- 有四条英文测试数据只列出三个选项，但答案索引指向缺失的第四个选项，这些数据在加载时会被跳过。另有六条英文数据
  包含三个有效选项，会作为三选项题目保留

## 评估说明

- 默认配置在 `test` 划分上进行 **0-shot** 评估，以准确率作为指标
- Few-shot 示例取自 `validation` 划分，英文和 10 个原生文字印度语言子集均提供该划分；`_roman` 子集没有 `validation` 划分，并会拒绝 few-shot 评估
- Sarvam 报告的零样本结果使用“Indic Average”：即 10 个原生文字印度语言子集的平均值，不包括英文子集和罗马化子集。该口径用于与 Sarvam 的结果进行比较，不是本适配器额外计算的聚合指标
- 使用 `subset_list` 评估指定的语言或文字（例如 `['hi', 'hi_roman']`）


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmlu_indic` |
| **数据集ID** | [sarvamai/mmlu-indic](https://modelscope.cn/datasets/sarvamai/mmlu-indic/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认样本数** | 0-shot |
| **评估分割** | `test` |
| **训练分割** | `validation` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 293,179 |
| 提示词长度（平均） | 606.06 字符 |
| 提示词长度（最小/最大） | 211 / 8204 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `en` | 14,038 | 648.16 | 223 | 5050 |
| `bn` | 14,042 | 582.21 | 223 | 6027 |
| `gu` | 14,042 | 578.53 | 223 | 6000 |
| `hi` | 14,042 | 592.55 | 223 | 6007 |
| `kn` | 14,042 | 615.7 | 223 | 8204 |
| `ml` | 14,042 | 610.76 | 223 | 6054 |
| `mr` | 14,042 | 585.12 | 224 | 7089 |
| `or` | 14,042 | 555.12 | 223 | 6004 |
| `pa` | 14,042 | 593.54 | 223 | 6012 |
| `ta` | 14,042 | 653.79 | 223 | 6394 |
| `te` | 14,042 | 598.43 | 223 | 5989 |
| `bn_roman` | 13,894 | 572.01 | 211 | 3562 |
| `gu_roman` | 14,004 | 592.19 | 211 | 6557 |
| `hi_roman` | 13,913 | 612.57 | 211 | 5139 |
| `kn_roman` | 14,005 | 632.05 | 211 | 4740 |
| `ml_roman` | 13,991 | 643.88 | 216 | 7117 |
| `mr_roman` | 13,904 | 595.36 | 211 | 4486 |
| `or_roman` | 13,979 | 602.7 | 211 | 4536 |
| `pa_roman` | 13,946 | 602.95 | 211 | 4622 |
| `ta_roman` | 13,096 | 639.27 | 216 | 3895 |
| `te_roman` | 13,989 | 622.46 | 211 | 5617 |

## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "b2910eca",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nUnlike most other early civilizations, Minoan culture shows little evidence of:\n\nA) trade.\nB) warfare.\nC) the development of a common religion.\nD) conspicuous consumption by elites."
    }
  ],
  "choices": [
    "trade.",
    "warfare.",
    "the development of a common religion.",
    "conspicuous consumption by elites."
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "language": "English"
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
    --datasets mmlu_indic \
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
    datasets=['mmlu_indic'],
    dataset_args={
        'mmlu_indic': {
            # subset_list: ['en', 'bn', 'gu']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
