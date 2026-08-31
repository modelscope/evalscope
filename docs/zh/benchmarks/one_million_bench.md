# $OneMillion-Bench


## 概述

$OneMillion-Bench（简称 $1M-Bench）用于评估语言模型和智能体在完成具有经济价值的专家级专业工作方面的表现。公开版本包含由金融、医疗、工业、法律和自然科学等领域的专家编写并审核的400个双语任务。

## 任务描述

- **任务类型**：基于评分标准的开放式专业问答，由大语言模型（LLM）进行评判
- **输入**：一段现实且上下文丰富的中文或英文专业请求
- **输出**：一份完整的自由格式专业分析报告或交付成果
- **领域**：经济学与金融、医疗与医学、工业、法律、自然科学

## 核心特性

- 包含400个零样本任务，在中文和全球两条语言赛道及五个专业领域之间均衡分布（每个语言-领域子集包含40个任务）
- 每个任务包含11至37条由专家编写的评分标准，涵盖事实信息、分析推理、指令遵循以及结构与格式等方面
- 每个任务同时包含正向评分项和负向扣分项，在公开版本中评分权重范围为 -20 到 12
- 样本以十个语言-领域子集的形式提供，以便在 EvalScope 报告中清晰展示论文中的语言赛道和领域划分

## 评估说明

- 需要使用 LLM 作为评判器。请配置 `judge.strategy='llm'`（或 `'auto'`）及 `judge.models`；官方评测框架当前推荐使用 Gemini 3.1 Pro Preview，但评判器的选择会影响绝对得分，此处未硬编码指定
- 对单个回答的所有评分标准将在一次请求中统一评判，并采用官方提供的二元命中/未命中（hit/miss）判断指令
- `expert_score` 是命中评分项的加权总和除以所有正向权重之和，结果裁剪至 `[0, 1]` 区间；当 `expert_score >= 0.7` 时，`pass_rate` 为 1，否则为 0
- 评判器回复必须包含每条评分标准恰好一次。格式错误的回复和传输失败将被排除，而非静默转换为零分
- 官方研究分别比较了基础模型、支持搜索的模型和深度研究型智能体。本原生适配器执行的是基准测试中的单轮生成路径；只有当外部工具调用型智能体的最终回答在相同评判器配置下进行评估时，其结果才具有可比性
- 任务通常要求生成长篇、带引用的报告。请配置足够大的生成和评判 `max_tokens` 值；使用一个评判器和一次重复运行时，完整评测将执行400次生成调用和400次评判调用

资源链接：[论文](https://arxiv.org/abs/2603.07980) |
[GitHub](https://github.com/humanlaya/OneMillion-Bench) |
[数据集](https://modelscope.cn/datasets/evalscope/OneMillion-Bench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `one_million_bench` |
| **数据集ID** | [evalscope/OneMillion-Bench](https://modelscope.cn/datasets/evalscope/OneMillion-Bench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2603.07980) |
| **标签** | `Agent`, `Knowledge`, `MultiLingual`, `QA`, `Reasoning` |
| **指标** | `expert_score`, `pass_rate` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 400 |
| 提示词长度（平均） | 1470.49 字符 |
| 提示词长度（最小/最大） | 105 / 15951 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示词平均长度 | 提示词最小长度 | 提示词最大长度 |
|--------|---------|-------------|------------|------------|
| `global_economics_and_finance` | 40 | 1810.97 | 868 | 3723 |
| `global_healthcare_and_medicine` | 40 | 2029.97 | 479 | 8870 |
| `global_industry` | 40 | 2581.15 | 453 | 9538 |
| `global_law` | 40 | 3277.78 | 404 | 15951 |
| `global_natural_sciences` | 40 | 1634.65 | 284 | 5775 |
| `cn_economics_and_finance` | 40 | 555.05 | 111 | 1169 |
| `cn_healthcare_and_medicine` | 40 | 584.38 | 135 | 2628 |
| `cn_industry` | 40 | 965.6 | 144 | 7755 |
| `cn_law` | 40 | 709.55 | 208 | 2041 |
| `cn_natural_sciences` | 40 | 555.83 | 105 | 1590 |

## 样例示例

**子集**: `global_economics_and_finance`

```json
{
  "input": [
    {
      "id": "24273bb5",
      "content": "You are an international financial risk analyst. Based on the Financial Stability Report released by the Bank of England's Financial Policy Committee in October 2025, global financial markets may face a \"risk of sharp market correction\" if in ... [TRUNCATED 924 chars] ... ields, and global capital flows.\nSpecial Conditions: Use only information published before December 31, 2025; do not fabricate information; generated content must cite real URLs. The answer must be complete and useful; do not fake a response."
    }
  ],
  "target": "[{\"rubric_number\": 1, \"rubric_detail\": \"Mention the high weighting of top US companies (e.g., Top 5 or AI-related stocks) in the index (approximately 30% or more) and identify this as a source of systemic risk.\", \"rubric_weight\": 10, \"rubric_ ... [TRUNCATED 3986 chars] ... c_number\": 16, \"rubric_detail\": \"The report contains hollow concluding remarks (e.g., \\\"In summary,\\\" \\\"We look forward to\\\") or transitional sentences lacking substantive content.\", \"rubric_weight\": -2, \"rubric_tag\": \"Analytical Reasoning\"}]",
  "id": 0,
  "group_id": 0,
  "subset_key": "global_economics_and_finance",
  "metadata": {
    "id": "e1b94c86-b6c9-43f6-8251-2e513e5efc52",
    "case_id": 1663,
    "language": "global",
    "domain": "Economics and Finance",
    "topics": [
      "Economics and Finance",
      "Financing & M&A",
      "Mergers & Acquisitions"
    ],
    "time_sensitivity": {
      "time_sensitivity": "Weakly time-sensitive",
      "year_month": "2025-10",
      "day": "NA"
    },
    "question": "You are an international financial risk analyst. Based on the Financial Stability Report released by the Bank of England's Financial Policy Committee in October 2025, global financial markets may face a \"risk of sharp market correction\" if in ... [TRUNCATED 924 chars] ... ields, and global capital flows.\nSpecial Conditions: Use only information published before December 31, 2025; do not fabricate information; generated content must cite real URLs. The answer must be complete and useful; do not fake a response."
  }
}
```

*注：部分内容因显示需要已被截断。*

## 提示模板

**提示模板：**
```text
{question}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets one_million_bench \
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
    datasets=['one_million_bench'],
    dataset_args={
        'one_million_bench': {
            # subset_list: ['global_economics_and_finance', 'global_healthcare_and_medicine', 'global_industry']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
