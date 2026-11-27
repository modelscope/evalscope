# Global-MMLU


## 概述

Global-MMLU 评估模型在多语言知识方面的表现，涵盖文化敏感和文化中立的问题。

## 任务描述

- **任务类型**：多语言多项选择题问答
- **输入**：一个 MMLU 问题及其四个选项，使用 42 种语言中的一种
- **输出**：正确选项的字母（A-D）
- **领域**：MMLU 的 57 个学科

## 主要特点

- 包含 42 种语言的 589,764 道测试题。每种语言有 2,850 道带有文化标签的题目。
- 保留了 39 行缺失选项的数据，其中包括 12 行正确答案为空的情况。

## 评估说明

- 在可用情况下，使用该语言 `dev` 数据集中同一学科的五个示例。
- 通过精确匹配对生成的答案字母进行评分。上游框架会对选项的可能性进行评分，其 Lite 任务按学科类别选择示例。
- 报告每种语言的整体准确率、文化敏感准确率和文化中立即准确率。Bayes@N 仅报告整体准确率。
- 采用 [Apache 2.0](https://opensource.org/license/apache-2-0) 许可证授权。


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `global_mmlu` |
| **数据集ID** | [CohereLabs/Global-MMLU](https://modelscope.cn/datasets/CohereLabs/Global-MMLU/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2412.03304) |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 5-shot |
| **评估划分** | `test` |
| **训练划分** | `dev` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 589,764 |
| 提示词长度（平均值） | 3048.64 字符 |
| 提示词长度（最小/最大值） | 558 / 16835 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示词平均值 | 提示词最小值 | 提示词最大值 |
|--------|---------|-------------|------------|------------|
| `am` | 14,042 | 2143.94 | 749 | 8937 |
| `ar` | 14,042 | 2718.89 | 830 | 11584 |
| `bn` | 14,042 | 2985.86 | 850 | 13433 |
| `cs` | 14,042 | 2955.66 | 863 | 13127 |
| `de` | 14,042 | 3570.76 | 1007 | 16685 |
| `el` | 14,042 | 3446.37 | 1003 | 15859 |
| `en` | 14,042 | 3178.85 | 949 | 14594 |
| `es` | 14,042 | 3469.11 | 977 | 15556 |
| `fa` | 14,042 | 2772.79 | 821 | 12018 |
| `fil` | 14,042 | 3641.24 | 997 | 16835 |
| `fr` | 14,042 | 3554.29 | 1015 | 15946 |
| `ha` | 14,042 | 3183.13 | 890 | 14093 |
| `he` | 14,042 | 2411.77 | 813 | 10445 |
| `hi` | 14,042 | 3087.54 | 900 | 13884 |
| `id` | 14,042 | 3447.94 | 938 | 15651 |
| `ig` | 14,042 | 2998.41 | 896 | 13562 |
| `it` | 14,042 | 3482.2 | 984 | 15316 |
| `ja` | 14,042 | 1571.26 | 627 | 6229 |
| `ko` | 14,042 | 1733.23 | 661 | 7096 |
| `ky` | 14,042 | 3121.07 | 936 | 14341 |
| `lt` | 14,042 | 3053.71 | 921 | 13664 |
| `mg` | 14,042 | 3567.95 | 1042 | 16250 |
| `ms` | 14,042 | 3345.47 | 947 | 15393 |
| `ne` | 14,042 | 2893.76 | 861 | 13033 |
| `nl` | 14,042 | 3488.92 | 993 | 16074 |
| `ny` | 14,042 | 3307.43 | 1012 | 15235 |
| `pl` | 14,042 | 3276.01 | 909 | 14503 |
| `pt` | 14,042 | 3295.65 | 970 | 14620 |
| `ro` | 14,042 | 3350.91 | 932 | 14854 |
| `ru` | 14,042 | 3319.02 | 920 | 14635 |
| `si` | 14,042 | 2958.06 | 881 | 13125 |
| `sn` | 14,042 | 3217.21 | 962 | 14630 |
| `so` | 14,042 | 3444.59 | 944 | 15603 |
| `sr` | 14,042 | 2976.99 | 901 | 13118 |
| `sv` | 14,042 | 3116.67 | 924 | 14506 |
| `sw` | 14,042 | 3214.23 | 941 | 14343 |
| `te` | 14,042 | 3044.29 | 900 | 13672 |
| `tr` | 14,042 | 3094.77 | 904 | 14003 |
| `uk` | 14,042 | 3112.33 | 908 | 13695 |
| `vi` | 14,042 | 3253.18 | 977 | 15261 |
| `yo` | 14,042 | 3002.93 | 938 | 13722 |
| `zh` | 14,042 | 1234.39 | 558 | 4397 |

## 样例示例

**子集**: `am`

```json
{
  "input": [
    {
      "id": "767e9bb5",
      "content": "Here are some examples of how to answer similar questions:\n\nZ_3[x]/(x^2 + c) መስክ እንደሆነ ሁሉ በZ_3 ውስጥ ያግኙ።\nA) 0\nB) 1\nC) 2\nD) 3\nANSWER: B\n\nመግለጫ 1 | aH የአንድ ፋክተር ቡድን ኤለመንት ከሆነ |aH| ይከፋፍላል |a|. መግለጫ 2 | H እና K የጂ ንዑስ ቡድን ከሆኑ HK የጂ ንዑስ ቡድን ነው።\nA) እው ... [TRUNCATED 469 chars] ... le choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nየተሰጠውን የመስክ ማራዘሚያ Q(sqrt(2)፣ sqrt(3)፣ sqrt(18)) ከQ በላይ አግኝ።\n\nA) 0\nB) 4\nC) 2\nD) 6"
    }
  ],
  "choices": [
    "0",
    "4",
    "2",
    "6"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "sample_id": "abstract_algebra/test/0",
    "subject": "abstract_algebra",
    "subject_category": "STEM",
    "cultural_sensitivity_label": "-",
    "language": "Amharic"
  }
}
```

*注意：为显示目的，部分内容已截断。*

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
    --datasets global_mmlu \
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
    datasets=['global_mmlu'],
    dataset_args={
        'global_mmlu': {
            # subset_list: ['am', 'ar', 'bn']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
