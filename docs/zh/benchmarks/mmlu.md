# MMLU


## 概述

MMLU（Massive Multitask Language Understanding，大规模多任务语言理解）是一个综合性评估基准，旨在衡量模型在预训练阶段所获得的知识。它涵盖 STEM、人文学科、社会科学及其他领域的 57 个学科，难度从基础到专业级别不等。

## 任务描述

- **任务类型**：多项选择题问答（Multiple-Choice Question Answering）
- **输入**：包含四个选项（A、B、C、D）的问题
- **输出**：单个正确答案的字母
- **学科范围**：57 个学科，分为 4 个类别（STEM、人文学科、社会科学、其他）

## 主要特点

- 覆盖从基础到高级专业水平的多样化知识领域
- 同时考察事实性知识和推理能力
- 包含抽象代数、解剖学、天文学、商业伦理等多个学科
- 是衡量大语言模型知识广度的标准基准

## 评估说明

- 默认配置使用开发集（dev split）中的 **5-shot** 示例
- 支持思维链（Chain-of-Thought, CoT）提示以提升推理能力
- 结果可按学科或类别（STEM、人文学科、社会科学、其他）进行聚合
- 使用 `subset_list` 参数可评估特定学科

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmlu` |
| **数据集 ID** | [cais/mmlu](https://modelscope.cn/datasets/cais/mmlu/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数量** | 5-shot |
| **评估集** | `test` |
| **训练集** | `dev` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 14,042 |
| 提示词长度（平均） | 3212.2 字符 |
| 提示词长度（最小/最大） | 985 / 14626 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `abstract_algebra` | 100 | 1256.98 | 1143 | 1383 |
| `anatomy` | 135 | 1446 | 1306 | 1783 |
| `astronomy` | 152 | 2616.64 | 2401 | 3191 |
| `business_ethics` | 100 | 2756.36 | 2492 | 3145 |
| `clinical_knowledge` | 265 | 1680.76 | 1510 | 1995 |
| `college_biology` | 144 | 2104.53 | 1874 | 2641 |
| `college_chemistry` | 100 | 1800.19 | 1641 | 2239 |
| `college_computer_science` | 100 | 3424.74 | 3129 | 4137 |
| `college_mathematics` | 100 | 1972.77 | 1776 | 2230 |
| `college_medicine` | 173 | 2377.62 | 2005 | 6779 |
| `college_physics` | 102 | 1941.26 | 1757 | 2237 |
| `computer_security` | 100 | 1598.89 | 1414 | 2238 |
| `conceptual_physics` | 235 | 1340.77 | 1246 | 1572 |
| `econometrics` | 114 | 2286.2 | 1976 | 2708 |
| `electrical_engineering` | 145 | 1372.1 | 1279 | 1578 |
| `elementary_mathematics` | 378 | 1856.81 | 1724 | 2322 |
| `formal_logic` | 126 | 2338.07 | 2065 | 3022 |
| `global_facts` | 100 | 1644.77 | 1558 | 2001 |
| `high_school_biology` | 310 | 2218.56 | 1971 | 2737 |
| `high_school_chemistry` | 203 | 1740.83 | 1519 | 2341 |
| `high_school_computer_science` | 100 | 3595.34 | 3224 | 4732 |
| `high_school_european_history` | 165 | 13421.12 | 12392 | 14626 |
| `high_school_geography` | 198 | 1849.07 | 1726 | 2204 |
| `high_school_government_and_politics` | 193 | 2355.29 | 2171 | 2922 |
| `high_school_macroeconomics` | 390 | 1861.47 | 1649 | 2206 |
| `high_school_mathematics` | 270 | 1731.31 | 1591 | 2224 |
| `high_school_microeconomics` | 238 | 1849.97 | 1651 | 2396 |
| `high_school_physics` | 151 | 2110.63 | 1836 | 2926 |
| `high_school_psychology` | 545 | 2431.39 | 2223 | 3498 |
| `high_school_statistics` | 216 | 3273.79 | 2932 | 4328 |
| `high_school_us_history` | 204 | 10530.61 | 9656 | 11469 |
| `high_school_world_history` | 237 | 6700.7 | 5650 | 8814 |
| `human_aging` | 223 | 1448.65 | 1335 | 1725 |
| `human_sexuality` | 131 | 1556.02 | 1412 | 2250 |
| `international_law` | 121 | 3094.31 | 2778 | 3432 |
| `jurisprudence` | 108 | 1851.24 | 1638 | 2370 |
| `logical_fallacies` | 163 | 2114.39 | 1932 | 2503 |
| `machine_learning` | 112 | 2852.01 | 2659 | 3150 |
| `management` | 103 | 1326.07 | 1220 | 1571 |
| `marketing` | 234 | 1984.28 | 1832 | 2266 |
| `medical_genetics` | 100 | 1531.32 | 1409 | 1775 |
| `miscellaneous` | 783 | 1121.57 | 1004 | 2096 |
| `moral_disputes` | 346 | 2300.58 | 2101 | 2711 |
| `moral_scenarios` | 895 | 2709.89 | 2644 | 2853 |
| `nutrition` | 306 | 2620.9 | 2408 | 3111 |
| `philosophy` | 311 | 1472.67 | 1319 | 2292 |
| `prehistory` | 324 | 2388.29 | 2201 | 2899 |
| `professional_accounting` | 282 | 2820.72 | 2515 | 3400 |
| `professional_law` | 1,534 | 8077.0 | 6997 | 10539 |
| `professional_medicine` | 272 | 4832.72 | 4380 | 5802 |
| `professional_psychology` | 612 | 2860.73 | 2594 | 3789 |
| `public_relations` | 110 | 1991.35 | 1824 | 2712 |
| `security_studies` | 245 | 6405.04 | 5680 | 7818 |
| `sociology` | 201 | 2176.49 | 1976 | 2530 |
| `us_foreign_policy` | 100 | 2129.28 | 1944 | 2393 |
| `virology` | 166 | 1563.11 | 1426 | 2507 |
| `world_religions` | 171 | 1051.68 | 985 | 1255 |

## 样例示例

**子集**: `abstract_algebra`

```json
{
  "input": [
    {
      "id": "c7cbfbb9",
      "content": "Here are some examples of how to answer similar questions:\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA) 0\nB) 1\nC) 2\nD) 3\nANSWER: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H ... [TRUNCATED] ... d be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n\nA) 0\nB) 4\nC) 2\nD) 6"
    }
  ],
  "choices": [
    "0",
    "4",
    "2",
    "6"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "abstract_algebra",
  "metadata": {
    "subject": "abstract_algebra"
  }
}
```

*注：部分内容为显示目的已截断。*

## 提示模板

**提示模板：**
```text
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmlu \
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
    datasets=['mmlu'],
    dataset_args={
        'mmlu': {
            # subset_list: ['abstract_algebra', 'anatomy', 'astronomy']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```