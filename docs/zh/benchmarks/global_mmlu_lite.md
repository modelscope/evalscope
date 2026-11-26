# Global-MMLU-Lite


## 概述

Global-MMLU-Lite（Cohere Labs）是 Global-MMLU 中均衡且全部经过人工翻译或后期编辑的子集：23 种语言中每种语言各有 200 道文化敏感（CS）和 200 道文化无关（CA）的 MMLU 题目。

## 任务描述

- **任务类型**：多语言多项选择问答
- **输入**：一道 23 种语言之一的 MMLU 题目及四个选项
- **输出**：正确选项的字母（A-D）
- **领域**：MMLU 57 个学科中的 43 个

## 主要特点

- 23 个子集，每种语言一个，各含 400 道测试题（实际评估共 9,199 道，见下文）以及 215 道 `dev` 题目（捷克语为 285 道），每个学科五道
- 语言集合与 Global-MMLU 不同：威尔士语、匈牙利语、缅甸语、奥里亚语、斯洛伐克语、阿尔巴尼亚语和塔吉克语仅出现在 Lite 版本中。论文中的 Lite 结果仅覆盖首次发布的 15 种语言
- 每道题都标注为 CS 或 CA
- 缅甸语（`my`）测试集第 202 行发布时缺少题干，因此被跳过，缅甸语共 399 道题
- 匈牙利语数据中的学科名称为匈牙利语；少样本示例改为按每行都带有的 `sample_id` 中的英文学科名称进行匹配
- 威尔士语（`cy`）发布时没有 `dev` 划分，因此始终以零样本方式评估

## 评估说明

- 默认配置使用 **5-shot** 评估，与论文一致。每道题的示例是同一语言中其所属学科的五道 `dev` 题目（每个学科恰好五道）
- 每种语言报告全部题目上的准确率，并分别报告文化敏感（`cs`）和文化无关（`ca`）题目上的准确率，作为准确率指标的 `cultural_sensitivity` 维度
- 各语言按数据集卡片给出的资源等级分组（`high-resource`、`mid-resource`、`low-resource`）
- 许可证：[Apache 2.0](https://opensource.org/license/apache-2-0)




## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `global_mmlu_lite` |
| **数据集ID** | [CohereLabs/Global-MMLU-Lite](https://modelscope.cn/datasets/CohereLabs/Global-MMLU-Lite/summary) |
| **论文** | [论文](https://arxiv.org/abs/2412.03304) |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认样本数** | 5-shot |
| **评估分割** | `test` |
| **训练分割** | `dev` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 9,199 |
| 提示词长度（平均） | 2616.93 字符 |
| 提示词长度（最小/最大） | 252 / 16179 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `ar` | 400 | 2469.9 | 995 | 11150 |
| `bn` | 400 | 2687.86 | 987 | 12826 |
| `cs` | 400 | 2680.84 | 958 | 12785 |
| `cy` | 400 | 615.88 | 252 | 2783 |
| `de` | 400 | 3215.65 | 1037 | 16179 |
| `en` | 400 | 2873.45 | 984 | 14330 |
| `es` | 400 | 3127.28 | 1019 | 15071 |
| `fr` | 400 | 3207.98 | 1037 | 15429 |
| `hi` | 400 | 2789.78 | 966 | 13386 |
| `hu` | 400 | 2978.34 | 996 | 14592 |
| `id` | 400 | 3073.09 | 1030 | 15137 |
| `it` | 400 | 3115.3 | 1024 | 15079 |
| `ja` | 400 | 1444.82 | 718 | 6013 |
| `ko` | 400 | 1571.25 | 709 | 6645 |
| `my` | 399 | 3147.42 | 1019 | 15746 |
| `or` | 400 | 2726.38 | 995 | 13442 |
| `pt` | 400 | 2974.54 | 994 | 14253 |
| `sk` | 400 | 2772.8 | 977 | 13133 |
| `sq` | 400 | 3030.3 | 1033 | 14570 |
| `sw` | 400 | 2882.9 | 1046 | 14042 |
| `tg` | 400 | 2955.62 | 981 | 13788 |
| `yo` | 400 | 2697.83 | 984 | 13569 |
| `zh` | 400 | 1151.58 | 624 | 4255 |

## 样例示例

**子集**: `ar`

```json
{
  "input": [
    {
      "id": "e4c86fc6",
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
