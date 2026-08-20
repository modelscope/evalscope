# Sanskriti


## 概述

Sanskriti 是一个多项选择题基准测试，用于评估对印度各邦文化、历史和地理知识的掌握情况。题目基于各邦特有的属性（如艺术、美食、节日等）构建，答案均参考维基百科。该适配器加载的是 SANSKRITI 论文（arXiv:2506.15355）中所用数据集在 ModelScope 上的镜像版本 `evalscope/Sanskriti`。

## 任务描述

- **任务类型**：多项选择题问答（Multiple-Choice Trivia Question Answering）
- **输入**：一道关于特定印度邦文化/地理/历史的问题，附带 4 个选项
- **输出**：正确答案对应的字母（A/B/C/D）
- **子集**：`association`（邦与属性关联类题目）、`country`（国家层面常识题）、`gk`（通用知识题）、`states`（邦识别类题目）

## 评估说明

- 默认配置采用 **0-shot** 评估方式（尽管上游数据集中该部分被命名为 `train`，但实际为评估数据）
- 问题和选项均为英文
- 论文指出部分题目涉及的文化元素存在模糊性；加载时会跳过少量（约 0.6%）答案字段与四个选项均不匹配的样本

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `sanskriti` |
| **数据集ID** | [evalscope/Sanskriti](https://modelscope.cn/datasets/evalscope/Sanskriti/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 21,726 |
| 提示词长度（平均） | 322.93 字符 |
| 提示词长度（最小/最大） | 256 / 636 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `association` | 5,453 | 343.41 | 273 | 523 |
| `country` | 5,563 | 284.48 | 256 | 417 |
| `gk` | 5,328 | 346.94 | 263 | 547 |
| `states` | 5,382 | 318.17 | 260 | 636 |

## 样例示例

**子集**: `association`

```json
{
  "input": [
    {
      "id": "0629b222",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nWhich of the given regions is home to the Jarawa body painting?\n\nA) Surguja district\nB) South Andaman and Middle Andaman Islands\nC) Buddha Marg, Patna\nD) Telangana"
    }
  ],
  "choices": [
    "Surguja district",
    "South Andaman and Middle Andaman Islands",
    "Buddha Marg, Patna",
    "Telangana"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "association",
  "metadata": {
    "state": "Andaman_and_Nicobar",
    "attribute": "Art"
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
    --datasets sanskriti \
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
    datasets=['sanskriti'],
    dataset_args={
        'sanskriti': {
            # subset_list: ['association', 'country', 'gk']  # 可选，用于指定评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
