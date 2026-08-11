# HiPhO


## 概述

HiPhO 是首个专注于高中物理奥林匹克竞赛并采用人类对齐评估的基准测试。它汇集了13场近期（2024-2025年）的国际及区域性奥赛试题，涵盖多种模态，包括纯文本题目和基于图表的题目。

## 任务描述

- **任务类型**：自由形式的物理问题求解，依据官方评分标准进行评分
- **输入**：一道物理题（包含常数表、上下文和问题），可选附带图表
- **输出**：分步解答，最终答案用 `<answer>...</answer>` 包裹并放入 `\boxed{}` 中
- **模态**：纯文本 和 文本+图表（示意图 / 变量图 / 数据图）

## 核心特性

- 共计403道题目，来自14套试卷（IPhO、APhO、EuPhO、NBPhO、PanPhO、PanMechanics、CPhO、F=MA），每套试卷作为一个独立子集。
- 英语试卷使用英文提示，中文试卷（CPhO、PanMechanics）使用中文提示，遵循官方语言对应关系。
- 复现论文中的两种评分机制，按题目分别应用：
  - **步骤级评分**：适用于提供官方评分细则的题目，LLM 评委对每个评分项打分，总分为各项得分之和。
  - **答案级评分**：适用于无评分细则的题目，通过基于规则的数学检查匹配 `\boxed{}` 中的最终答案；若规则检查失败，则回退至 LLM 评委判断。

## 评估说明

- 需要 LLM 评委：运行时需设置 `judge_strategy='llm'`（或 `'auto'`，该选项会为此基准自动启用评委），并提供 `judge_model_args`。不支持 `judge_strategy='rule'`。
- 主要指标：`accuracy`，即每道题得分与满分之比（范围 `[0, 1]`），按子集取平均值聚合。
  - 对于步骤级题目，满分为所有评分项分数之和；
  - 对于拥有多个官方评分方案的题目（如 EuPhO、NBPhO），采用得分最高的方案，与论文一致。
- 报告的是每场考试的标准化得分，**不计算**论文中金/银/铜牌的分数线，因为这需要原始总分和官方截止分数。
- 解答可能很长，且图表题需要视觉输入；请为被测模型设置较大的 `generation_config.max_tokens`。若解答在 `<answer>` 块前被截断，则无法提取答案，得分接近零，但这与物理能力无关。
- 图表以内联 base64 形式发送，最大约 1.5 MB；若所用模型对单图有更小限制，请在 `dataset_args` 中设置 `max_image_bytes`。
- 资源链接：[论文](https://arxiv.org/abs/2509.07894) | [GitHub](https://github.com/SciYu/HiPhO) | [排行榜](https://phyarena.github.io/)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `hipho` |
| **数据集ID** | [evalscope/HiPhO](https://modelscope.cn/datasets/evalscope/HiPhO/summary) |
| **论文** | [Paper](https://arxiv.org/abs/2509.07894) |
| **标签** | `Math`, `MultiModal`, `QA`, `Reasoning` |
| **指标** | `accuracy` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 403 |
| 提示词长度（平均） | 3020.35 字符 |
| 提示词长度（最小/最大） | 653 / 9336 字符 |

**各子集统计：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `APhO_2025` | 45 | 4624.02 | 2496 | 8787 |
| `CPhO_2025` | 43 | 2041.81 | 960 | 3745 |
| `EuPhO_2024` | 7 | 1924.29 | 1468 | 2051 |
| `EuPhO_2025` | 6 | 1646.33 | 1422 | 1856 |
| `F=MA_2024` | 25 | 1598.76 | 1279 | 1957 |
| `F=MA_2025` | 25 | 1721.2 | 1395 | 2513 |
| `IPhO_2024` | 37 | 4152.57 | 2201 | 6701 |
| `IPhO_2025` | 39 | 6359.74 | 3362 | 9336 |
| `NBPhO_2024` | 24 | 2305.25 | 1317 | 4486 |
| `NBPhO_2025` | 20 | 2677.7 | 1359 | 4808 |
| `PanMechanics_2024` | 29 | 878.55 | 653 | 1283 |
| `PanMechanics_2025` | 23 | 874.87 | 667 | 1150 |
| `PanPhO_2024` | 33 | 2820.55 | 1448 | 3880 |
| `PanPhO_2025` | 47 | 3526.47 | 1561 | 6209 |

**图像统计：**

| 指标 | 值 |
|--------|-------|
| 图像总数 | 413 |
| 每样本图像数 | 最小: 1, 最大: 5, 平均: 1.5 |
| 分辨率范围 | 456x60 - 3200x1645 |
| 格式 | png |


## 样例示例

**子集**: `APhO_2025`

```json
{
  "input": [
    {
      "id": "b441ea36",
      "content": [
        {
          "text": "You are participating in a high school physics Olympiad exam.\nPlease read the following question carefully and provide a clear, step-by-step solution with full reasoning.\nInstructions:\n1. Use LaTeX to format all variables, equations, and calc ... [TRUNCATED 3334 chars] ... gamma} R^{\\delta}$ \nwhere $G$ is the gravitational constant, and $\\beta, \\gamma$ and $\\delta$ are constant exponents.\nQuestion (Answer only the question stated below):\nFind the values of exponents: (1) $\\beta$, (2) $\\gamma$, and (3) $\\delta$."
        },
        {
          "image": "[BASE64_IMAGE: png, ~101.8KB]"
        }
      ]
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "APhO_2025",
  "metadata": {
    "id": "APhO_2025_1_A_1",
    "source": "APhO_2025",
    "question": "Find the values of exponents: (1) $\\beta$, (2) $\\gamma$, and (3) $\\delta$.",
    "answers": [
      "\\boxed{$\\beta = 2$}",
      "\\boxed{$\\gamma = -1$}",
      "\\boxed{$\\delta = 4$}"
    ],
    "marking": [
      [
        "Award 0.2 pt if the answer correctly expresses the dimension of $G$ as $[G] = L^3 M^{-1} T^{-2}$, where $L$ is the base dimensions length, $M$ is mass, and $T$ is time. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer correctly sets up the exponent equation $0 = 2 - \\beta$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer correctly sets up the exponent equation $0 = \\gamma + 1$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer correctly sets up the exponent equation $1 = \\delta - 3$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer obtains the correct value $\\beta = 2$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer obtains the correct value $\\gamma = -1$. Otherwise, award 0 pt.",
        "Award 0.1 pt if the answer obtains the correct value $\\delta = 4$. Otherwise, award 0 pt."
      ]
    ]
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
    --datasets hipho \
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
    datasets=['hipho'],
    dataset_args={
        'hipho': {
            # subset_list: ['APhO_2025', 'CPhO_2025', 'EuPhO_2024']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
