# ProcessBench

## 概述

ProcessBench 是一个用于评估 AI 模型在数学推理过程验证能力的基准测试。它测试模型在从 GSM8K 到 OmniMath 等不同难度级别的数学问题中，识别分步解答错误的能力。

## 任务描述

- **任务类型**：数学推理错误检测
- **输入**：数学问题 + 分步解答（带标签的段落）
- **输出**：第一个错误段落的索引（若解答正确则为 -1）
- **领域**：数学推理验证、错误检测

## 主要特性

- 包含四个难度子集：
  - `gsm8k`：小学数学问题
  - `math`：竞赛数学问题
  - `olympiadbench`：奥林匹克级别问题
  - `omnimath`：高级数学推理
- 测试过程监督与验证能力
- 要求逐段分析推理过程以发现错误

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 跟踪多项指标：
  - `error_acc`：检测错误位置的准确率
  - `correct_acc`：识别正确解答的准确率
  - `simple_f1_score`：综合两项的 F1 分数
- 答案应使用 \boxed{} 格式（段落索引或 -1）
- 聚合方法：**F1** 分数

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `process_bench` |
| **数据集 ID** | [Qwen/ProcessBench](https://modelscope.cn/datasets/Qwen/ProcessBench/summary) |
| **论文** | N/A |
| **标签** | `Math`, `Reasoning` |
| **指标** | `error_acc`, `correct_acc`, `simple_f1_score` |
| **默认示例数量** | 0-shot |
| **评估划分** | `test` |
| **聚合方式** | `f1` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 3,400 |
| 提示词长度（平均） | 2764.83 字符 |
| 提示词长度（最小/最大） | 690 / 9005 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `gsm8k` | 400 | 1824.26 | 876 | 4520 |
| `math` | 1,000 | 2297.11 | 690 | 7565 |
| `olympiadbench` | 1,000 | 3166.77 | 1129 | 9005 |
| `omnimath` | 1,000 | 3206.82 | 832 | 8550 |

## 样例示例

**子集**: `gsm8k`

```json
{
  "input": [
    {
      "id": "aca63163",
      "content": "The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):\n\n[Math Problem]\n\nSue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, t ... [TRUNCATED] ... nce you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes \"not found\").\n\nPlease put your final answer (i.e., the index) in \boxed{}.\n"
    }
  ],
  "target": "1",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "steps": [
      "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
      "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, \\(1/3 \\times 18 = 6\\) flamingos are taken back. So, they have \\(18 - 6 = 12\\) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has \\(12 + 6 = 18\\) pink flamingos and 6 white flamingos.",
      "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has \\(18 + 18 = 36\\) pink flamingos and still 6 white flamingos.",
      "To find the difference, subtract the number of white flamingos from the number of pink flamingos: \\(36 - 6 = 30\\). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is \\(\\boxed{30}\\)."
    ],
    "tagged_response": "<paragraph_0>\nTo find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.\n</paragrap ... [TRUNCATED] ...  subtract the number of white flamingos from the number of pink flamingos: \\(36 - 6 = 30\\). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is \\(\\boxed{30}\\).\n</paragraph_3>",
    "final_answer_correct": false
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \boxed{}.

```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets process_bench \
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
    datasets=['process_bench'],
    dataset_args={
        'process_bench': {
            # subset_list: ['gsm8k', 'math', 'olympiadbench']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```