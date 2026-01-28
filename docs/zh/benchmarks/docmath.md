# DocMath

## 概述

DocMath-Eval 是一个专注于特定领域内数值推理的综合性基准测试。该基准要求模型理解长篇且专业的文档，并基于文档内容进行数值推理以回答问题。

## 任务描述

- **任务类型**：基于文档的数学推理
- **输入**：长文档上下文 + 数值推理问题
- **输出**：带推理过程的数值答案
- **重点**：长上下文理解与定量推理能力

## 主要特点

- 需要理解长篇专业文档
- 在文档上下文中进行数值推理
- 多种复杂度级别（复杂/简单，长/短）
- 测试真实场景下的文档理解能力
- 同时考察阅读理解与数学技能

## 评估说明

- 默认配置使用 **0-shot** 评估
- 使用 LLM-as-judge 进行答案评估
- 子集包括：`complong_testmini`、`compshort_testmini`、`simplong_testmini`、`simpshort_testmini`
- 答案格式：`"Therefore, the answer is (answer)"`

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `docmath` |
| **数据集ID** | [yale-nlp/DocMath-Eval](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary) |
| **论文** | N/A |
| **标签** | `LongContext`, `Math`, `Reasoning` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 800 |
| 提示词长度（平均） | 68791.03 字符 |
| 提示词长度（最小/最大） | 505 / 1009038 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `complong_testmini` | 300 | 175355.17 | 18687 | 1009038 |
| `compshort_testmini` | 200 | 1990.74 | 505 | 9460 |
| `simplong_testmini` | 100 | 13972.84 | 6870 | 24001 |
| `simpshort_testmini` | 200 | 3154.2 | 560 | 9600 |

## 样例示例

**子集**: `complong_testmini`

```json
{
  "input": [
    {
      "id": "a07cbfcf",
      "content": "Please read the following text and answer the question below.\n\n<text>\nDELTA AIR LINES, INC.\nConsolidated Balance Sheets\n| (in millions, except share data) | March 31, 2018 | December 31, 2017 |\n| ASSETS |\n| Current Assets: |\n| Cash and cash e ... [TRUNCATED] ... comprehensive income for foreign currency exchange contracts in 2017 and 2018, and the changes in value for derivative contracts and other in 2018, in million?\n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
    }
  ],
  "target": "-31.0",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "complong-testmini-0",
    "answer_type": "float"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets docmath \
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
    datasets=['docmath'],
    dataset_args={
        'docmath': {
            # subset_list: ['complong_testmini', 'compshort_testmini', 'simplong_testmini']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```