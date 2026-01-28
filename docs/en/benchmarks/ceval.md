# C-Eval


## Overview

C-Eval is a comprehensive Chinese evaluation benchmark designed to assess the knowledge and reasoning abilities of language models in Chinese. It covers 52 subjects ranging from STEM to humanities and social sciences, with questions from middle school to professional examination levels.

## Task Description

- **Task Type**: Multiple-Choice Question Answering (Chinese)
- **Input**: Chinese question with four answer choices (A, B, C, D)
- **Output**: Single correct answer letter
- **Subjects**: 52 subjects organized into 4 categories (STEM, Social Science, Humanities, Other)

## Key Features

- 13,948 multiple-choice questions across 52 subjects
- Questions sourced from Chinese middle school, high school, college, and professional exams
- Covers diverse domains including mathematics, physics, law, medicine, and more
- Includes explanations for validation split questions
- Standard benchmark for Chinese language model evaluation

## Evaluation Notes

- Default configuration uses **5-shot** examples from the dev split
- Questions and prompts are in Chinese
- Answers should follow the format: "答案：[LETTER]"
- Results can be aggregated by subject or category
- Use `subset_list` parameter to evaluate specific subjects


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ceval` |
| **Dataset ID** | [evalscope/ceval](https://modelscope.cn/datasets/evalscope/ceval/summary) |
| **Paper** | N/A |
| **Tags** | `Chinese`, `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 5-shot |
| **Evaluation Split** | `val` |
| **Train Split** | `dev` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,346 |
| Prompt Length (Mean) | 1643.61 chars |
| Prompt Length (Min/Max) | 727 / 6605 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `computer_network` | 19 | 1245.42 | 1201 | 1313 |
| `operating_system` | 19 | 1216.16 | 1187 | 1282 |
| `computer_architecture` | 21 | 1654.67 | 1622 | 1732 |
| `college_programming` | 37 | 1745.03 | 1660 | 2189 |
| `college_physics` | 19 | 2071.16 | 1986 | 2184 |
| `college_chemistry` | 24 | 2152.96 | 2107 | 2291 |
| `advanced_mathematics` | 19 | 6271.68 | 6130 | 6605 |
| `probability_and_statistics` | 18 | 4700.39 | 4527 | 4987 |
| `discrete_mathematics` | 16 | 1176.75 | 1104 | 1365 |
| `electrical_engineer` | 37 | 1137.81 | 1093 | 1264 |
| `metrology_engineer` | 24 | 1187.08 | 1140 | 1312 |
| `high_school_mathematics` | 18 | 2749.11 | 2670 | 2894 |
| `high_school_physics` | 19 | 1365 | 1263 | 1538 |
| `high_school_chemistry` | 19 | 1625.79 | 1536 | 1739 |
| `high_school_biology` | 19 | 1159.95 | 1103 | 1243 |
| `middle_school_mathematics` | 19 | 2141.68 | 2054 | 2413 |
| `middle_school_biology` | 21 | 1822.52 | 1769 | 1910 |
| `middle_school_physics` | 19 | 1596.53 | 1544 | 1721 |
| `middle_school_chemistry` | 20 | 1916.65 | 1857 | 2045 |
| `veterinary_medicine` | 23 | 1254.04 | 1210 | 1350 |
| `college_economics` | 55 | 1919.85 | 1863 | 2141 |
| `business_administration` | 33 | 1608.52 | 1547 | 1754 |
| `marxism` | 19 | 1061.32 | 1033 | 1104 |
| `mao_zedong_thought` | 24 | 1485.62 | 1449 | 1545 |
| `education_science` | 29 | 1369.24 | 1338 | 1444 |
| `teacher_qualification` | 44 | 1516.55 | 1457 | 1638 |
| `high_school_politics` | 19 | 2062 | 1955 | 2189 |
| `high_school_geography` | 19 | 1059.53 | 1026 | 1216 |
| `middle_school_politics` | 21 | 1653.24 | 1595 | 1714 |
| `middle_school_geography` | 12 | 1063.58 | 1020 | 1143 |
| `modern_chinese_history` | 23 | 1355.7 | 1313 | 1447 |
| `ideological_and_moral_cultivation` | 19 | 760.21 | 727 | 830 |
| `logic` | 22 | 2436.41 | 2358 | 2572 |
| `law` | 24 | 1799.29 | 1729 | 1931 |
| `chinese_language_and_literature` | 23 | 954.83 | 937 | 983 |
| `art_studies` | 33 | 793.3 | 774 | 844 |
| `professional_tour_guide` | 29 | 924.41 | 902 | 1004 |
| `legal_professional` | 23 | 2856.17 | 2718 | 2978 |
| `high_school_chinese` | 19 | 2295.79 | 2205 | 2418 |
| `high_school_history` | 20 | 1221.9 | 1164 | 1300 |
| `middle_school_history` | 22 | 1069.73 | 1034 | 1149 |
| `civil_servant` | 47 | 1973 | 1849 | 2186 |
| `sports_science` | 19 | 1810.26 | 1789 | 1874 |
| `plant_protection` | 22 | 1678.09 | 1653 | 1745 |
| `basic_medicine` | 19 | 938.05 | 920 | 976 |
| `clinical_medicine` | 22 | 1119.41 | 1086 | 1209 |
| `urban_and_rural_planner` | 46 | 1428.8 | 1373 | 1591 |
| `accountant` | 49 | 1605.92 | 1511 | 1808 |
| `fire_engineer` | 31 | 1240.81 | 1168 | 1402 |
| `environmental_impact_assessment_engineer` | 31 | 1269.97 | 1209 | 1388 |
| `tax_accountant` | 49 | 1970.65 | 1879 | 2099 |
| `physician` | 49 | 1010.59 | 983 | 1065 |

## Sample Example

**Subset**: `computer_network`

```json
{
  "input": [
    {
      "id": "73073a35",
      "content": "以下是一些示例问题：\n\n问题：下列设备属于资源子网的是____。\n选项：\nA. 计算机软件\nB. 网桥\nC. 交换机\nD. 路由器\n解析：1. 首先，资源子网是指提供共享资源的网络，如打印机、文件服务器等。\r\n2. 其次，我们需要了解选项中设备的功能。网桥、交换机和路由器的主要功能是实现不同网络之间的通信和数据传输，是通信子网设备。而计算机软件可以提供共享资源的功能。\n答案：A\n\n问题：滑动窗口的作用是____。\n选项：\nA. 流量控制\nB. 拥塞控制\nC. 路由控制\nD. 差错 ... [TRUNCATED] ... Mbps，所以答案为min{80Mbps, 100Mbps}=80Mbps，选C。\n答案：C\n\n\n以下是中国关于计算机网络的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：\"答案：[LETTER]\"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。\n\n问题：使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____\n选项：\nA. 1\nB. 2\nC. 3\nD. 4\n"
    }
  ],
  "choices": [
    "1",
    "2",
    "3",
    "4"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "id": 0,
    "explanation": "",
    "subject": "computer_network"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
以下是中国关于{subject}的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。

问题：{question}
选项：
{choices}

```

<details>
<summary>Few-shot Template</summary>

```text
以下是一些示例问题：

{fewshot}


```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ceval \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['ceval'],
    dataset_args={
        'ceval': {
            # subset_list: ['computer_network', 'operating_system', 'computer_architecture']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


