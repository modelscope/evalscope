# Chinese-SimpleQA


## 概述

Chinese SimpleQA 是一个中文问答数据集，旨在评估语言模型在简单事实性问题上的表现。该数据集测试模型在不同知识领域中理解和生成正确中文答案的能力。

## 任务描述

- **任务类型**：中文事实性问答
- **输入**：中文简单事实性问题
- **输出**：中文事实性答案
- **语言**：中文

## 主要特点

- 覆盖多个知识领域的多样化主题
- 测试世界知识的简单事实性问题
- 中文语言能力评估
- 使用 LLM-as-judge 评估答案正确性
- 提供多个类别子集

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用 LLM-as-judge 进行评估
- 指标：`is_correct`、`is_incorrect`、`is_not_attempted`
- 评估事实准确性，不要求完全匹配

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `chinese_simpleqa` |
| **数据集ID** | [AI-ModelScope/Chinese-SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary) |
| **论文** | N/A |
| **标签** | `Chinese`, `Knowledge`, `QA` |
| **指标** | `is_correct`, `is_incorrect`, `is_not_attempted` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,000 |
| 提示词长度（平均） | 32.45 字符 |
| 提示词长度（最小/最大） | 16 / 129 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `中华文化` | 326 | 32.09 | 18 | 86 |
| `人文与社会科学` | 609 | 33.94 | 18 | 87 |
| `工程、技术与应用科学` | 481 | 33.13 | 18 | 91 |
| `生活、艺术与文化` | 601 | 32.4 | 17 | 76 |
| `社会` | 453 | 32.33 | 18 | 129 |
| `自然与自然科学` | 530 | 30.49 | 16 | 83 |

## 样例示例

**子集**: `中华文化`

```json
{
  "input": [
    {
      "id": "b6b48177",
      "content": "请回答问题：\n\n伏兔穴所属的经脉是什么？"
    }
  ],
  "target": "足阳明胃经",
  "id": 0,
  "group_id": 0,
  "subset_key": "中华文化",
  "metadata": {
    "id": "97e7f58a3b154facaa3a5c64d678c7bf",
    "primary_category": "中华文化",
    "secondary_category": "中医"
  }
}
```

## 提示模板

**提示模板：**
```text
请回答问题：

{question}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets chinese_simpleqa \
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
    datasets=['chinese_simpleqa'],
    dataset_args={
        'chinese_simpleqa': {
            # subset_list: ['中华文化', '人文与社会科学', '工程、技术与应用科学']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```