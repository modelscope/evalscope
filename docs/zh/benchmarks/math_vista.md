# MathVista

## 概述

MathVista 是一个面向视觉上下文中的数学推理的综合性基准测试。它结合了新创建的数据集与现有基准，用于评估模型在多个领域中处理多样化视觉数学推理任务的能力。

## 任务描述

- **任务类型**：视觉数学推理  
- **输入**：包含数学问题的图像（选择题或多选题/自由作答）  
- **输出**：数值答案或选项字母  
- **领域**：几何、代数、统计学、科学推理  

## 主要特点

- 包含来自 31 个不同数据集的 6,141 个示例  
- 包含新构建的数据集 IQTest、FunctionQA 和 PaperQA  
- 整合了文献中的 9 个 MathQA 数据集和 19 个 VQA 数据集  
- 测试对谜题图形的逻辑推理能力  
- 测试对函数图像的代数推理能力  
- 测试对学术论文图表的科学推理能力  

## 评估说明

- 默认配置使用 **0-shot** 方式在 `testmini` 子集上进行评估  
- 支持选择题和自由作答题两种题型  
- 答案应以 `\boxed{}` 格式给出，且不带单位  
- 使用数值等价性检查进行答案比对  
- 对选择题采用思维链（Chain-of-Thought, CoT）提示方法  

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `math_vista` |
| **数据集 ID** | [evalscope/MathVista](https://modelscope.cn/datasets/evalscope/MathVista/summary) |
| **论文** | N/A |
| **标签** | `MCQ`, `Math`, `MultiModal`, `Reasoning` |
| **指标** | `acc` |
| **默认样本数（Shots）** | 0-shot |
| **评估子集** | `testmini` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,000 |
| 提示词长度（平均） | 261.48 字符 |
| 提示词长度（最小/最大） | 106 / 1391 字符 |

**图像统计信息：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 1,000 |
| 每样本图像数 | 最小: 1, 最大: 1, 平均: 1 |
| 分辨率范围 | 187x18 - 5236x3491 |
| 图像格式 | jpeg, mpo, png, webp |

## 样例示例

**子集**：`default`

```json
{
  "input": [
    {
      "id": "93ec4f16",
      "content": [
        {
          "text": "When a spring does work on an object, we cannot find the work by simply multiplying the spring force by the object's displacement. The reason is that there is no one value for the force-it changes. However, we can split the displacement up in ... [TRUNCATED] ... g of spring constant $k=750 \\mathrm{~N} / \\mathrm{m}$. When the canister is momentarily stopped by the spring, by what distance $d$ is the spring compressed?\nPlease reason step by step, and put your final answer within \\boxed{} without units."
        },
        {
          "image": "[BASE64_IMAGE: jpg, ~185.7KB]"
        }
      ]
    }
  ],
  "target": "1.2",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "precision": 1.0,
    "question_type": "free_form",
    "answer_type": "float",
    "category": "math-targeted-vqa",
    "context": "scientific figure",
    "grade": "college",
    "img_height": 720,
    "img_width": 1514,
    "language": "english",
    "skills": [
      "scientific reasoning"
    ],
    "source": "SciBench",
    "split": "testmini",
    "task": "textbook question answering"
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}} without units.
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets math_vista \
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
    datasets=['math_vista'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```