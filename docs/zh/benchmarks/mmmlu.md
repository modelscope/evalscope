# MMMLU


## 概述

MMMLU（Multilingual Massive Multitask Language Understanding，多语言大规模多任务语言理解）是 MMLU 基准测试的多语言扩展版本。它在 14 种语言中评估语言模型的多语言知识与推理能力，涵盖原始 MMLU 基准中的 57 个学科。

## 任务描述

- **任务类型**：多语言多项选择题问答
- **输入**：以 14 种语言之一呈现的问题，包含四个选项（A、B、C、D）
- **输出**：单个正确答案字母
- **语言**：阿拉伯语、孟加拉语、德语、西班牙语、法语、印地语、印尼语、意大利语、日语、韩语、葡萄牙语、斯瓦希里语、约鲁巴语、中文
- **学科**：MMLU 中的 57 个学科（STEM、人文学科、社会科学、其他）

## 主要特点

- 完整 MMLU 基准的多语言翻译
- 覆盖主要语系的 14 种类型学上多样化的语言
- 测试跨语言知识迁移与多语言推理能力
- 与原始 MMLU 相同的学科覆盖范围（57 个学科）
- 包含低资源语言（例如斯瓦希里语、约鲁巴语）

## 评估说明

- 默认配置使用 **0-shot** 评估（仅测试集）
- 使用 `subset_list` 评估特定语言（例如 `['ZH_CN', 'JA_JP', 'FR_FR']`）
- 结果按语言子集分组
- 支持跨语言性能比较

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmmlu` |
| **数据集ID** | [openai-mirror/MMMLU](https://modelscope.cn/datasets/openai-mirror/MMMLU/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ`, `MultiLingual` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 196,588 |
| 提示词长度（平均） | 624.75 字符 |
| 提示词长度（最小/最大） | 136 / 5975 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `AR_XY` | 14,042 | 584.94 | 231 | 4735 |
| `BN_BD` | 14,042 | 654.99 | 247 | 4914 |
| `DE_DE` | 14,042 | 791.64 | 294 | 5657 |
| `ES_LA` | 14,042 | 753.18 | 271 | 5791 |
| `FR_FR` | 14,042 | 777.82 | 278 | 5952 |
| `HI_IN` | 14,042 | 675.02 | 256 | 5379 |
| `ID_ID` | 14,042 | 726.51 | 270 | 5539 |
| `IT_IT` | 14,042 | 761.19 | 277 | 5975 |
| `JA_JP` | 14,042 | 322.79 | 149 | 2064 |
| `KO_KR` | 14,042 | 354.35 | 153 | 2345 |
| `PT_BR` | 14,042 | 706.79 | 258 | 5635 |
| `SW_KE` | 14,042 | 699.08 | 259 | 5566 |
| `YO_NG` | 14,042 | 681.01 | 248 | 5644 |
| `ZH_CN` | 14,042 | 257.15 | 136 | 1495 |

## 样例示例

**子集**: `AR_XY`

```json
{
  "input": [
    {
      "id": "e43faf14",
      "content": "أجب على سؤال الاختيار من متعدد التالي. يجب أن يكون السطر الأخير من إجابتك بالتنسيق التالي: 'ANSWER: [LETTER]' (بدون علامات اقتباس) حيث [LETTER] هو أحد الحروف A,B,C,D. فكّر خطوة بخطوة قبل الإجابة.\n\nأوجد درجة امتداد الحقل المحدد Q(sqrt(2)، sqrt(3)، sqrt(18)) على Q.\n\nA) 0\nB) 4\nC) 2\nD) 6"
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
    "subject": "abstract_algebra",
    "language": "AR_XY"
  }
}
```

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

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
    --datasets mmmlu \
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
    datasets=['mmmlu'],
    dataset_args={
        'mmmlu': {
            # subset_list: ['AR_XY', 'BN_BD', 'DE_DE']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```