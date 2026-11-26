# Global-MMLU

## 概述

Global-MMLU（Cohere Labs，2024）是覆盖 42 种语言的 MMLU。它结合了机器翻译、专业翻译和社区后期编辑，并将部分题目标注为文化敏感（CS）或文化无关（CA），从而区分模型的多语言知识与以西方为中心的知识。

## 任务描述

- **任务类型**：多语言多项选择问答
- **输入**：一道 42 种语言之一的 MMLU 题目及四个选项
- **输出**：正确选项的字母（A-D）
- **领域**：MMLU 的 57 个学科

## 主要特点

- 42 个子集，每种语言一个，各含全部 14,042 道 MMLU 测试题（共计 589,764 道）以及 285 道 `dev` 题目（每个学科五道）
- 每种语言有 2,850 道题带有文化敏感性标签：792 道 CS、2,058 道 CA，其余未标注
- 39 道测试题（0.007%，主要为阿姆哈拉语、印地语和越南语）在翻译中丢失了一个选项；与 lm-evaluation-harness 一致，这些题目保留并将该选项显示为空，因此每种语言都保留全部 14,042 道题。其中 12 道丢失的恰好是正确选项
- 如需 23 种语言、全部人工翻译的较小题集，请使用 `global_mmlu_lite`

## 评估说明

- 默认配置使用 **5-shot** 评估，与论文一致。每道题的示例是同一语言中其所属学科的五道 `dev` 题目（每个学科恰好五道）
- 每种语言报告全部题目上的准确率，并分别报告文化敏感（`cs`）和文化无关（`ca`）题目上的准确率，作为准确率指标的 `cultural_sensitivity` 维度
- 各语言按数据集卡片给出的资源等级分组（`high-resource`、`mid-resource`、`low-resource`）
- 许可证：[Apache 2.0](https://opensource.org/license/apache-2-0)

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `global_mmlu` |
| **数据集ID** | [CohereLabs/Global-MMLU](https://modelscope.cn/datasets/CohereLabs/Global-MMLU/summary) |
| **论文** | [论文](https://arxiv.org/abs/2412.03304) |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认样本数** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `dev` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 589,764 |
| 提示词长度（平均） | 3049.29 字符 |
| 提示词长度（最小/最大） | 558 / 16835 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `am` | 14,042 | 2144.97 | 749 | 8937 |
| `ar` | 14,042 | 2718.9 | 830 | 11584 |
| `bn` | 14,042 | 2986.99 | 850 | 13433 |
| `cs` | 14,042 | 2956.69 | 863 | 13127 |
| `de` | 14,042 | 3570.78 | 1007 | 16685 |
| `el` | 14,042 | 3447.4 | 1003 | 15859 |
| `en` | 14,042 | 3179.89 | 949 | 14594 |
| `es` | 14,042 | 3469.11 | 977 | 15556 |
| `fa` | 14,042 | 2773.82 | 821 | 12018 |
| `fil` | 14,042 | 3642.27 | 997 | 16835 |
| `fr` | 14,042 | 3554.32 | 1015 | 15947 |
| `ha` | 14,042 | 3184.16 | 890 | 14093 |
| `he` | 14,042 | 2412.8 | 813 | 10445 |
| `hi` | 14,042 | 3087.61 | 900 | 13884 |
| `id` | 14,042 | 3447.94 | 938 | 15651 |
| `ig` | 14,042 | 2999.44 | 896 | 13562 |
| `it` | 14,042 | 3482.26 | 984 | 15316 |
| `ja` | 14,042 | 1571.26 | 627 | 6229 |
| `ko` | 14,042 | 1733.25 | 661 | 7097 |
| `ky` | 14,042 | 3122.1 | 936 | 14341 |
| `lt` | 14,042 | 3054.74 | 921 | 13664 |
| `mg` | 14,042 | 3568.98 | 1042 | 16250 |
| `ms` | 14,042 | 3346.5 | 947 | 15393 |
| `ne` | 14,042 | 2894.82 | 861 | 13033 |
| `nl` | 14,042 | 3488.92 | 993 | 16074 |
| `ny` | 14,042 | 3308.46 | 1012 | 15235 |
| `pl` | 14,042 | 3276.01 | 909 | 14503 |
| `pt` | 14,042 | 3295.69 | 970 | 14620 |
| `ro` | 14,042 | 3351.94 | 932 | 14854 |
| `ru` | 14,042 | 3319.02 | 920 | 14635 |
| `si` | 14,042 | 2959.09 | 881 | 13125 |
| `sn` | 14,042 | 3218.24 | 962 | 14630 |
| `so` | 14,042 | 3445.62 | 944 | 15603 |
| `sr` | 14,042 | 2978.02 | 901 | 13118 |
| `sv` | 14,042 | 3117.7 | 924 | 14506 |
| `sw` | 14,042 | 3215.31 | 941 | 14343 |
| `te` | 14,042 | 3045.32 | 900 | 13672 |
| `tr` | 14,042 | 3094.77 | 904 | 14003 |
| `uk` | 14,042 | 3113.36 | 908 | 13695 |
| `vi` | 14,042 | 3253.18 | 977 | 15261 |
| `yo` | 14,042 | 3004.3 | 938 | 13722 |
| `zh` | 14,042 | 1234.4 | 558 | 4397 |

## 样例示例

**子集**: `am`

```json
{
  "input": [
    {
      "id": "c45ee91e",
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

*注：部分内容已截断显示。*

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
