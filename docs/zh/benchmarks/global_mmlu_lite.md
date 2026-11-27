# Global-MMLU-Lite


## 概述

Global-MMLU-Lite 在 23 种经人工审核的语言中，平衡了文化敏感型和文化中立型问题。

## 任务描述

- **任务类型**：多语言选择题问答
- **输入**：以 23 种语言之一呈现的包含四个选项的 MMLU 问题
- **输出**：正确选项的字母（A-D）
- **领域**：MMLU 57 个学科中的 43 个

## 主要特点

- 版本 3.0 在跳过一个空白的缅甸语问题后，评估了 9,199 个问题。论文涵盖了前 15 种语言。
- 威尔士语使用零样本（zero-shot），因为其发布版本没有 `dev` 划分。

## 评估说明

- 如果可用，则从该语言的 `dev` 划分中选取五个同主题示例。
- 通过精确匹配对生成的答案字母进行评分。上游测试框架对选项可能性进行评分，其 Lite 任务按主题类别选择示例。
- 报告每种语言的整体准确率、文化敏感型准确率和文化中立型准确率。Bayes@N 仅报告整体准确率。
- 采用 [Apache 2.0](https://opensource.org/license/apache-2-0) 许可证授权。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `global_mmlu_lite` |
| **数据集ID** | [CohereLabs/Global-MMLU-Lite](https://modelscope.cn/datasets/CohereLabs/Global-MMLU-Lite/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2412.03304) |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `dev` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 9,199 |
| 提示词长度（平均值） | 2616.82 字符 |
| 提示词长度（最小/最大） | 252 / 16179 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `ar` | 400 | 2469.9 | 995 | 11150 |
| `bn` | 400 | 2687.55 | 987 | 12825 |
| `cs` | 400 | 2680.58 | 958 | 12785 |
| `cy` | 400 | 615.85 | 252 | 2783 |
| `de` | 400 | 3215.62 | 1037 | 16179 |
| `en` | 400 | 2873.17 | 984 | 14330 |
| `es` | 400 | 3127.28 | 1019 | 15071 |
| `fr` | 400 | 3207.97 | 1037 | 15429 |
| `hi` | 400 | 2789.74 | 966 | 13386 |
| `hu` | 400 | 2978.19 | 996 | 14592 |
| `id` | 400 | 3073.08 | 1030 | 15137 |
| `it` | 400 | 3115.24 | 1024 | 15079 |
| `ja` | 400 | 1444.81 | 718 | 6013 |
| `ko` | 400 | 1571.24 | 709 | 6645 |
| `my` | 399 | 3147.32 | 1018 | 15746 |
| `or` | 400 | 2726.38 | 995 | 13442 |
| `pt` | 400 | 2974.53 | 994 | 14253 |
| `sk` | 400 | 2772.8 | 977 | 13133 |
| `sq` | 400 | 3030.12 | 1033 | 14570 |
| `sw` | 400 | 2882.58 | 1046 | 14042 |
| `tg` | 400 | 2955.62 | 981 | 13788 |
| `yo` | 400 | 2696.96 | 984 | 13569 |
| `zh` | 400 | 1151.58 | 624 | 4255 |

## 样例示例

**子集**: `ar`

```json
{
  "input": [
    {
      "id": "87416dc0",
      "content": "Here are some examples of how to answer similar questions:\n\nأنت تدفع شاحنة على طول الطريق. هل سيكون من الأسهل تسريع هذه الشاحنة على المريخ؟ لماذا؟ (افترض عدم وجود احتكاك)\nA) سيكون الأمر أصعب لأن الشاحنة أثقل وزنًا على المريخ.\nB) سيكون الأمر أ ... [TRUNCATED 1714 chars] ...  following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nعند السفر شمالًا من الولايات المتحدة إلى كندا، سترى نجم الشمال (بولاريس) _________.\n\nA) أكثر سطوعًا\nB) باهت\nC) أعلى في السماء\nD) أكثر انخفاضًا في السماء"
    }
  ],
  "choices": [
    "أكثر سطوعًا",
    "باهت",
    "أعلى في السماء",
    "أكثر انخفاضًا في السماء"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "sample_id": "astronomy/test/58",
    "subject": "astronomy",
    "subject_category": "STEM",
    "cultural_sensitivity_label": "CS",
    "language": "Arabic"
  }
}
```

*注意：部分内容为显示目的已截断。*

## 提示模板

**提示模板:**
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
    --datasets global_mmlu_lite \
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
    datasets=['global_mmlu_lite'],
    dataset_args={
        'global_mmlu_lite': {
            # subset_list: ['ar', 'bn', 'cs']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
