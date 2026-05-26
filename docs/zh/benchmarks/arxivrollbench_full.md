# ArxivRollBench-Full

## 概述

ArxivRollBench 是一个基于近期 arXiv 论文构建的滚动基准。`arxivrollbench_full` 提供完整公开划分，用于在所有可用样本上进行更全面的评估。

## 任务描述

- **任务类型**：科学文本多项选择推理
- **输入**：近期 arXiv 文本片段和四个候选答案
- **输出**：单个正确答案字母（A、B、C 或 D）
- **领域**：计算机科学、数量金融、数学、物理、统计、定量生物、经济学和电气工程与系统科学
- **版本**：2024b、2025a 和 2026a 滚动快照

## 主要特点

- 时间感知的基准快照有助于降低数据污染导致的能力高估
- 覆盖多个 arXiv 领域和科学写作风格
- 在 SCP 框架下包含排序、完形填空和预测三类格式
- `arxivrollbench_full` 使用完整公开划分，适合正式完整评测
- 成本可控的紧凑划分可通过 `arxivrollbench` 使用

## 评估说明

- 默认配置使用 **0-shot** 评估
- `arxivrollbench_full` 基准使用完整公开数据集
- 如需成本可控的紧凑划分，请使用 `arxivrollbench`
- 每个子集都会从 `liangzid` 命名空间下对应的公开 ModelScope 镜像加载
- 答案会标准化为 A-D，并使用准确率进行评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `arxivrollbench_full` |
| **数据集 ID** | [liangzid/arxivrollbench-full](https://modelscope.cn/datasets/liangzid/arxivrollbench-full/summary) |
| **论文** | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/41098) |
| **标签** | `Knowledge`, `MCQ`, `Reasoning` |
| **指标** | `acc` |
| **默认样本数** | 0-shot |
| **评估划分** | `train` |

## 数据统计

*暂无统计信息。*

## 样例示例

*暂无样例示例。*

## 提示模板

**提示模板：**
```text
Answer the following ArxivRollBench multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets arxivrollbench_full \
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
    datasets=['arxivrollbench_full'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
