# MGSM

## 概述

MGSM（Multilingual Grade School Math，多语言小学数学）是一个用于评估语言模型多语言数学推理能力的基准测试。它将 GSM8K 扩展至 11 种类型学上多样化的语言，以检验模型是否能在不同语言中执行思维链（chain-of-thought）推理。

## 任务描述

- **任务类型**：多语言数学应用题求解  
- **输入**：11 种语言之一的小学数学应用题  
- **输出**：包含逐步推理过程和数值答案的解答  
- **语言**：英语、西班牙语、法语、德语、俄语、中文、日语、泰语、斯瓦希里语、孟加拉语、泰卢固语  

## 主要特点

- 每种语言包含 250 道题目（从 GSM8K 翻译而来）  
- 覆盖 11 种类型学上多样化的语言，涵盖不同语系  
- 测试模型的多语言思维链推理能力  
- 各语言使用相同的问题内容，便于跨语言比较  
- 旨在评估与语言无关的数学推理能力  

## 评估说明

- 默认配置使用 **4-shot** 示例  
- 答案应使用 `\boxed{}` 格式包裹，以便正确提取  
- 可通过 `subset_list` 参数指定评估特定语言（例如 `['en', 'zh', 'ja']`）  
- 支持跨语言性能比较  
- 少样本（few-shot）示例从同语言的训练集（train split）中抽取  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mgsm` |
| **数据集ID** | [evalscope/mgsm](https://modelscope.cn/datasets/evalscope/mgsm/summary) |
| **论文** | N/A |
| **标签** | `Math`, `MultiLingual`, `Reasoning` |
| **指标** | `acc` |
| **默认少样本数量** | 4-shot |
| **评估集** | `test` |
| **训练集** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 2,750 |
| 提示词长度（平均） | 1742.98 字符 |
| 提示词长度（最小/最大） | 791 / 2464 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `en` | 250 | 1790.71 | 1637 | 2165 |
| `es` | 250 | 1940.02 | 1773 | 2371 |
| `fr` | 250 | 2047 | 1878 | 2440 |
| `de` | 250 | 1963.9 | 1792 | 2386 |
| `ru` | 250 | 1831.66 | 1667 | 2214 |
| `zh` | 250 | 842.16 | 791 | 946 |
| `ja` | 250 | 1102.33 | 1035 | 1248 |
| `th` | 250 | 1835.53 | 1699 | 2135 |
| `sw` | 250 | 1953.48 | 1780 | 2354 |
| `bn` | 250 | 1759.28 | 1601 | 2106 |
| `te` | 250 | 2106.77 | 1939 | 2464 |

## 样例示例

**子集**: `en`

```json
{
  "input": [
    {
      "id": "d67cb3cf",
      "content": "Here are some examples of how to solve similar problems:\n\nQuestion: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\n\nReasoning:\nStep-by-Step Answer: Roger sta ... [TRUNCATED] ...  every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\n"
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": null,
    "equation_solution": null
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{}.


```

<details>
<summary>少样本模板</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}

{question}
Please reason step by step, and put your final answer within \boxed{}.


```

</details>

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mgsm \
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
    datasets=['mgsm'],
    dataset_args={
        'mgsm': {
            # subset_list: ['en', 'es', 'fr']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```