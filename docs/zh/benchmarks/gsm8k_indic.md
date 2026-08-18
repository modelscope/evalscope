# GSM8K-Indic


## 概述

GSM8K-Indic 将 GSM8K 小学数学应用题翻译为 10 种印度语言，每种语言均提供原生文字版本和罗马化（拉丁转写）版本，同时还包含原始英文版本。

## 任务描述

- **任务类型**：多语言数学应用题求解
- **输入**：21 种语言/文字变体之一的小学数学应用题
- **输出**：通过逐步推理得出的数值答案
- **语言**：孟加拉语、英语、古吉拉特语、印地语、卡纳达语、马拉雅拉姆语、马拉地语、奥里亚语、旁遮普语、泰米尔语、泰卢固语 —— 每种印度语言均包含原生文字版本和 `_roman` 转写版本

## 评估说明

- 默认配置使用 **0-shot** 评估（仅提供测试集）
- 使用 `subset_list` 参数评估特定语言/文字（例如 `['hi', 'hi_roman']`）
- 标准答案为原始英文推理链的最终数值结果；仅问题部分被翻译

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `gsm8k_indic` |
| **数据集ID** | [sarvamai/gsm8k-indic](https://modelscope.cn/datasets/sarvamai/gsm8k-indic/summary) |
| **论文** | N/A |
| **标签** | `Math`, `MultiLingual`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 27,670 |
| 提示词长度（平均） | 335.91 字符 |
| 提示词长度（最小/最大） | 135 / 1045 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `en` | 1,319 | 310.87 | 144 | 919 |
| `bn` | 1,319 | 306.06 | 136 | 790 |
| `gu` | 1,319 | 303.06 | 135 | 770 |
| `hi` | 1,319 | 311.1 | 147 | 761 |
| `kn` | 1,319 | 338.11 | 145 | 831 |
| `ml` | 1,319 | 345.21 | 156 | 907 |
| `mr` | 1,319 | 315.01 | 146 | 785 |
| `or` | 1,319 | 313.28 | 141 | 815 |
| `pa` | 1,319 | 312.77 | 152 | 800 |
| `ta` | 1,319 | 367.12 | 161 | 972 |
| `te` | 1,319 | 330.78 | 142 | 877 |
| `bn_roman` | 1,319 | 330.69 | 153 | 816 |
| `gu_roman` | 1,318 | 335.73 | 148 | 885 |
| `hi_roman` | 1,319 | 340.77 | 154 | 869 |
| `kn_roman` | 1,316 | 368.49 | 149 | 959 |
| `ml_roman` | 1,319 | 363.39 | 159 | 937 |
| `mr_roman` | 1,310 | 339.56 | 156 | 848 |
| `or_roman` | 1,319 | 339.92 | 158 | 930 |
| `pa_roman` | 1,319 | 334.74 | 155 | 847 |
| `ta_roman` | 1,303 | 387.6 | 163 | 1045 |
| `te_roman` | 1,319 | 360.46 | 145 | 1025 |

## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "92205129",
      "content": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "target": "18",
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
{question}
Please reason step by step, and put your final answer within \boxed{{}}.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gsm8k_indic \
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
    datasets=['gsm8k_indic'],
    dataset_args={
        'gsm8k_indic': {
            # subset_list: ['en', 'bn', 'gu']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
