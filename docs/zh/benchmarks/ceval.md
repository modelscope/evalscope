# C-Eval


## 概述

C-Eval 是一个全面的中文评估基准，旨在评估语言模型在中文语境下的知识与推理能力。该基准涵盖 52 个学科，范围从 STEM（科学、技术、工程和数学）到人文与社会科学，题目难度覆盖初中至专业资格考试水平。

## 任务描述

- **任务类型**：多项选择题问答（中文）
- **输入**：一道包含四个选项（A、B、C、D）的中文问题
- **输出**：单个正确答案字母
- **学科分类**：52 个学科，分为 4 大类（STEM、社会科学、人文学科、其他）

## 主要特点

- 共计 13,948 道多选题，覆盖 52 个学科
- 题目来源包括中国初中、高中、大学及专业资格考试
- 涵盖数学、物理、法律、医学等多个领域
- 验证集（validation split）题目附带解析
- 是中文语言模型评估的标准基准之一

## 评估说明

- 默认配置使用来自开发集（dev split）的 **5-shot** 示例
- 所有问题和提示均为中文
- 答案格式应为："答案：[LETTER]"
- 结果可按学科或大类进行汇总
- 可通过 `subset_list` 参数指定评估特定学科

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `ceval` |
| **数据集ID** | [evalscope/ceval](https://modelscope.cn/datasets/evalscope/ceval/summary) |
| **论文** | N/A |
| **标签** | `Chinese`, `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 5-shot |
| **评估集** | `val` |
| **训练集** | `dev` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 1,346 |
| 提示词长度（平均） | 1643.61 字符 |
| 提示词长度（最小/最大） | 727 / 6605 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
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

## 样例示例

**子集**: `computer_network`

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

*注：部分内容因展示需要已被截断。*

## 提示模板

**提示模板：**
```text
以下是中国关于{subject}的单项选择题，请选出对的选项。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 A、B、C、D 中的一个。

问题：{question}
选项：
{choices}

```

<details>
<summary>少样本（Few-shot）模板</summary>

```text
以下是一些示例问题：

{fewshot}


```

</details>

## 使用方法

### 使用命令行（CLI）

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ceval \
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
    datasets=['ceval'],
    dataset_args={
        'ceval': {
            # subset_list: ['computer_network', 'operating_system', 'computer_architecture']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```