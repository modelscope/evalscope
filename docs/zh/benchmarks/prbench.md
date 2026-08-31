# PRBench


## 概述

PRBench（Professional Reasoning Benchmark，专业推理基准测试）评估模型在现实且高风险的金融与法律问题上的开放式推理能力。该基准测试由领域专家编写的对话和细粒度评分标准组成，用于衡量模型的回答是否准确、有用、可审计，并能恰当地处理不确定性与风险。

## 任务描述

- **任务类型**：基于评分标准的多轮开放式问答
- **输入**：1 到 10 轮对话，可选择性附带参考文本
- **输出**：助手对最后一轮用户提问的回复
- **领域**：金融与法律领域的专业推理

## 核心特性

- 当前版本包含 1,100 个对话和 18,692 条专家精心设计的评分标准，覆盖 13 个金融主题和 12 个法律主题；约 30% 的对话为多轮对话。
- 覆盖 114 个国家及地区以及美国 47 个司法管辖区，包含专家与非专家用户场景。
- 提供四个数据集划分：`finance`（600）、`legal`（500）、`finance_hard`（300）和 `legal_hard`（250）。其中“hard”划分包含各自完整划分中最难的样本。
- 每个样本包含 10–30 条独立评分的标准，每条标准具有 -10 到 10（不含 0）之间的整数权重。正权重标准描述期望的属性，负权重标准描述不期望的属性。

## 评估说明

- 每条评分标准均由 LLM 评判器根据官方提示独立判断是否满足。论文中使用 `o4-mini` 作为评判器；请配置 `judge.models` 并设置 `judge.strategy='auto'` 或 `'llm'`。
- `clipped_score` 是论文报告的主要指标：每个样本的加权得分除以其正权重总和，然后对所有样本取平均值，并将结果下限裁剪至 0。
- `normalized_score` 先减去每个样本可能的最低得分（即其负权重之和），再除以完整得分范围，最后对样本得分取平均。该指标适用于不同评分标准分布间的比较。
- 参考文本会严格按照官方评估器的方式，直接前置到对应的用户轮次内容之前。若评判器在解析或传输过程中失败，则排除受影响的样本而非赋予分数。
- 完整的金融加法律评估每次需调用评判器 18,692 次。由于“hard”划分与完整划分存在重叠，EvalScope 会独立报告每个划分的结果，不提供 `OVERALL` 行；发布综合结果时，请选择完整划分或 hard 划分之一。
- 相关资源：[论文](https://arxiv.org/abs/2511.11562) |
  [GitHub](https://github.com/scaleapi/PRBench) |
  [数据集](https://modelscope.cn/datasets/ScaleAI/PRBench)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `prbench` |
| **数据集ID** | [ScaleAI/PRBench](https://modelscope.cn/datasets/ScaleAI/PRBench/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2511.11562) |
| **标签** | `Knowledge`, `MultiTurn`, `QA`, `Reasoning` |
| **指标** | `clipped_score`, `normalized_score` |
| **默认示例数** | 0-shot |
| **评估划分** | `finance` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,650 |
| 提示词长度（平均） | 10251.59 字符 |
| 提示词长度（最小/最大） | 84 / 132961 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `finance` | 600 | 9590.54 | 103 | 132961 |
| `legal` | 500 | 10328.35 | 84 | 99212 |
| `finance_hard` | 300 | 10796.97 | 103 | 132961 |
| `legal_hard` | 250 | 11030.14 | 140 | 99212 |

## 样例示例

**子集**: `finance`

```json
{
  "input": [
    {
      "id": "5026042c",
      "content": "hey, we've got a problem with our credit loss forecasting model that feeds into CCAR stress testing. during validaton, we found the model is systematically underestimating losses in the tail scenarios by about 15-20%, particularly for our com ... [TRUNCATED 259 chars] ... . The model was performing fine until we started seeing these newer macro scenarios with simultaneous interest rate spikes and credit spread widening. thoughts on how to approach this? what would you prioritize given the timeline constraints?"
    },
    {
      "id": "59952e89",
      "content": "**Quick‑Response Blueprint for the CCAR Credit Loss Model (6‑Week Deadline)**  \n*(Tail losses are under‑forecasting by 15‑20 % in the commercial real‑estate (CRE) portfolio under the newest macro stress scenarios.)*\n\n| Phase | Timeframe | Wha ... [TRUNCATED 8109 chars] ... , and Legal.  \n\nIf you follow the table‑derived 6‑week schedule, you’ll either have a fully corrected model that meets CCAR standards **or** a defensible overlay that keeps you compliant until a full rebuild can be rolled out post‑submission."
    },
    {
      "id": "c0794365",
      "content": "this is really helpful but i'm concerned about the week 2 copula approach for the joint shock scenario. our quant team is pretty thin right now and honestly we've never implemented a copula-based stress scenario before. is there a simpler way to capture the simultaneous rate/spread shock that doesn't require us to build new statistical infrastructure in 1 week? maybe something more rule-based that regulators would still accept?"
    },
    {
      "id": "2312a13c",
      "content": "### Quick‑Rule Based Stress‑Scenario Design (No Copula Required)\n\n| What you’ll need | Why it’s regulator‑friendly | How much effort |\n|-------------------|-----------------------------|-----------------|\n| **A single, “joint‑shock” driver**  ... [TRUNCATED 5124 chars] ... n **2‑3 business days**.  \nIf you later decide to adopt a full copula, you’ll have the historical data ready, and at that point the switch will only be a code‑update, not a concept‑change.  \n\nGood luck – you’ve got this 6‑week window in hand!"
    },
    {
      "id": "c8b32de6",
      "content": "ok the linear blend approach makes sense, but i'm worried about using a fixed correlation of 0.6, that seems arbitrary. when I look at our historical data from 2020-2022, the coreelation between rate changes and our CRE spread changes was clo ... [TRUNCATED 55 chars] ... should we be using a rolling correlation or just pick the most conservative period? also, for the backtesting you mentioned, we don't have clean monthly loss data, our charge offs are reported quarterly. will that be a problem for validation?"
    },
    {
      "id": "269be9b4",
      "content": "## 1.  Choosing a Correlation Coefficient  \n| Issue | Why it matters | Practical options | How to document it |\n|-------|----------------|-------------------|--------------------|\n| **Historical correlation swings** (0.3 in 2008‑09, 0.8 in 20 ... [TRUNCATED 5868 chars] ... le, you have a defensible, regulator‑approved stress scenario that can be built into the CCAR submission in less than a week.  \n\nLet me know if you need a sample Excel template or a quick‑script for the correlation calculation—happy to share!"
    },
    {
      "id": "ccf03620",
      "content": "one more thing on the backtesting, when you say \"coverage check\" where all observed quarterly charge offs be less or equal to predicted loss of the 0.3 correlation scenario, what if we have a quarterly where actual charge offs exceeded even our most conservative prediction? this happened in Q2 2020 when we had some one off covid related writedowns that spiked our numbers, do we just document this as an outlier or does that break our whole validation approach?"
    },
    {
      "id": "cb010996",
      "content": "### The “Coverage Check” – What It’s Built For  \n\nIn CCAR, the coverage check is a very *high‑level* sanity test:  \n\n> “For every historical period we used in the back‑test, the probability‑weighted loss predicted by **our worst‑case** stress ... [TRUNCATED 6304 chars] ... nal).” |\n\nYou’re perfectly allowed to keep the linear‑blend approach, use the 0.8 correlation as core, and treat Q2 2020 as an exceptional event. That meets FRB guidance, keeps you on schedule, and preserves the integrity of your CCAR submit."
    },
    {
      "id": "d8d952ce",
      "content": "got it on the outlier treatment. but now i'm realizing we might need different correlation assumptions across our CRE sub-segments. our office portfolio behaves very differently from retail and multifamily during rate shocks, should we be using segment specific correlatiosn or is that going to make the documentation too complex for the regulators? also, do we need separate back testing for each segment or can we do it at the portfolio level?"
    },
    {
      "id": "f6976632",
      "content": "## 1.  Do you need sector‑specific correlations?\n\n| Factor | Why it *might* matter | When it *doesn’t* matter | Practical rule‑of‑thumb |\n|--------|-----------------------|--------------------------|------------------------|\n| **Collateral pr ... [TRUNCATED 9748 chars] ... more time proving the same logic. Stick to 3–4 meaningful sub‑portfolio categories, back‑test each, and use the portfolio‑level test as an overall guardrail.  \nThat should keep the submission robust, defendable, and time‑efficient. Good luck!"
    },
    "... [TRUNCATED 9 more items] ..."
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task": "ea67e314b6c2e8fc70627c19",
    "field": "Finance",
    "topic": "Risk Management & Stress Testing",
    "expert": "Expert",
    "turns": 10,
    "rubrics": [
      {
        "id": "f46d80f7-6c5e-47a4-93f7-b05e99b24c79",
        "title": "The response specifically mentions Federal Reserve SR 11-7 \"Guidance on Model Risk Management\" or references the three lines of defense framework for model governance when discussing overlay approval processes.",
        "weight": 9.0,
        "category": "Risk & Regulatory Disclosure"
      },
      {
        "id": "668345b3-2b95-4be7-8b49-3d5cf17fea28",
        "title": "The response addresses the requirement for \"effective challenge\" documentation per Federal Reserve guidance when discussing overlay methodologies.",
        "weight": 7.0,
        "category": "Risk & Regulatory Disclosure"
      },
      {
        "id": "a33da628-133b-4704-a894-6ba7b7bb3ad6",
        "title": "The response mentions specific CCAR submission requirements such as \"FR Y-14A Schedule H.1\" for stress testing results or references the Capital Plan Rule's documentation standards for model adjustments.",
        "weight": 6.0,
        "category": "Risk & Regulatory Disclosure"
      },
      {
        "id": "e00429ae-5d51-4dea-bddc-b9903effc0dd",
        "title": "The response discusses overlay \"procyclicality\" concerns or mentions the need to avoid overlays that amplify economic cycles, referencing Basel III buffer framework principles.",
        "weight": 6.0,
        "category": "Risk & Regulatory Disclosure"
      },
      {
        "id": "9a047436-c0de-4311-8971-26a0f5a5029f",
        "title": "The response specifies Model Risk Management (MRM) ticket numbers, change control procedures, or mentions the need for \"model development standards\" documentation per regulatory expectations.",
        "weight": 3.0,
        "category": "Risk & Regulatory Disclosure"
      },
      {
        "id": "e51fdf10-fb6b-46f0-adfb-bb1e152b0b1e",
        "title": "The response addresses \"out-of-sample testing\" requirements for overlay validation or mentions the need for \"holdout datasets\" when calibrating overlay factors.",
        "weight": 5.0,
        "category": "Process Transparency & Auditability"
      },
      {
        "id": "ff7fba39-49fc-4ffe-a37a-f733bbc2c196",
        "title": "The response mentions specific audit committee charter requirements such as \"Sarbanes-Oxley Section 404\" compliance for internal controls over financial reporting when discussing overlay approval authority.",
        "weight": 2.0,
        "category": "Risk & Regulatory Disclosure"
      },
      {
        "id": "955d276b-5bb4-4a9e-970b-8ad3465c8686",
        "title": "The response incorrectly suggests applying overlays to baseline scenarios or recommends overlays that affect both stressed and non-stressed projections simultaneously.",
        "weight": -6.0,
        "category": "Financial Accuracy"
      },
      {
        "id": "1910dc61-966a-4907-a3f7-6851d569d3c0",
        "title": "The response refers to the key factors for overlay approval by the audit committee, e.g., clear business rationale, quantitative justification, and governance controls.",
        "weight": 7.0,
        "category": "Risk & Regulatory Disclosure"
      },
      {
        "id": "0e29c49a-30e7-4a23-8589-d8a994452f67",
        "title": "The response discusses \"model conceptual soundness review\" requirements or mentions the need for independent quantitative validation of overlay assumptions per Federal Reserve SR 14-1 guidance.",
        "weight": 3.0,
        "category": "Risk & Regulatory Disclosure"
      },
      "... [TRUNCATED 7 more items] ..."
    ],
    "economic_pathway": "Compliance and Reporting Integrity, Risk & Resilience",
    "decision_type": "Modeling & Measurement, Compliance & Reporting"
  }
}
```

*注：部分内容因展示需要已被截断。*

## 提示模板

*未定义提示模板。*

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets prbench \
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
    datasets=['prbench'],
    dataset_args={
        'prbench': {
            # subset_list: ['finance', 'legal', 'finance_hard']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
