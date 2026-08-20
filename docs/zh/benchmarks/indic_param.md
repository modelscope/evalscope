# IndicParam


## 概述

IndicParam 是一个面向研究生水平的基准测试，用于评估大语言模型（LLM）对低资源及极低资源印度语系语言的理解能力。全部 13,207 道选择题均来自官方 UGC-NET 语言科目试题及答案，以各语言的原生文字（或梵语-英语混合形式）呈现。

## 任务描述

- **任务类型**：研究生水平多项选择题问答
- **输入**：一道用低资源印度语系语言编写的 UGC-NET 考试试题，包含 4 个选项
- **输出**：正确答案的字母（A/B/C/D）
- **语言**：博多语（Bodo）、多格里语（Dogri）、古吉拉特语（Surya 字体）、孔卡尼语（Konkani）、迈蒂利语（Maithili）、马拉地语（Marathi）、尼泊尔语（Nepali）、奥里亚语（Oriya）、拉贾斯坦语（Rajasthani）、梵语（Sanskrit）、梵语-英语混合语（Sanskrit-English code-mixed）、桑塔利语（Santali）

## 主要特点

- 包含 13,207 道来自官方 UGC-NET 语言科目试题的多项选择题
- 覆盖 12 种低资源印度语系语言/文字，包括博多语和桑塔利语等极低资源语言
- 所有问题均以对应语言的原生文字（或梵语-英语混合形式）呈现
- 所有语言数据统一打包在一个数据集配置中，通过 `subject` 字段区分不同语言

## 评估说明

- 默认配置采用 **0-shot** 评估方式（仅提供 test 分割）
- 可通过 `subset_list` 参数指定评估特定语言
- 所有语言数据统一打包在一个数据集配置中，通过 `subject` 字段区分；本适配器会按该字段重新组织数据格式

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `indic_param` |
| **数据集 ID** | [bharatgenai/IndicParam](https://modelscope.cn/datasets/bharatgenai/IndicParam/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 13,207 |
| 提示词长度（平均） | 376.02 字符 |
| 提示词长度（最小/最大） | 218 / 1413 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `Bodo` | 1,313 | 461.37 | 256 | 738 |
| `Dogri` | 1,027 | 487.72 | 245 | 853 |
| `Gujarati_surya` | 1,044 | 395.79 | 255 | 611 |
| `Konkani` | 1,328 | 396.77 | 245 | 1413 |
| `Maithili` | 1,286 | 284.67 | 218 | 451 |
| `Marathi` | 1,245 | 382.66 | 242 | 957 |
| `Nepali` | 1,038 | 406.12 | 260 | 857 |
| `Oriya` | 577 | 365.04 | 239 | 924 |
| `Rajasthani` | 1,190 | 321.32 | 237 | 1136 |
| `Sanskrit` | 1,315 | 304.51 | 229 | 833 |
| `Sanskrit Mix` | 971 | 352.41 | 253 | 693 |
| `Santali` | 873 | 366.16 | 233 | 809 |

## 样例示例

**子集**: `Bodo`

```json
{
  "input": [
    {
      "id": "0616580a",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nआथिखालाव सुबुं थुनलाइफोरखौ बुथुमनो थाखाय बबे आदबखौ रासिनै बाहायनाय जायो\n\nA) फट' दैखांनाय\nB) रेकरडिं खालामनाय\nC) सल बुंहोनाय\nD) सल खोनासंनाय"
    }
  ],
  "choices": [
    "फट' दैखांनाय",
    "रेकरडिं खालामनाय",
    "सल बुंहोनाय",
    "सल खोनासंनाय"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "Bodo",
  "metadata": {
    "subject": "Bodo",
    "exam_name": "Question Papers of NET Dec. 2012 Bodo Paper III hindi"
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
    --datasets indic_param \
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
    datasets=['indic_param'],
    dataset_args={
        'indic_param': {
            # subset_list: ['Bodo', 'Dogri', 'Gujarati_surya']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
