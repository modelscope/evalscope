# KINA


## 概述

KINA（Knowledge Index of Noah's Ark，诺亚方舟知识索引）是一个高密度、多学科的知识基准测试，用于评估大语言模型能否解答横跨261个细粒度学科的专家级问题。它是首个将“学科代表性”作为核心设计原则的基准测试。

## 任务描述

- **任务类型**：多项选择题问答（MCQ）
- **输入**：一个特定学科的问题，附带最多10个字母选项（A–J）
- **输出**：单个正确答案字母（A–J）
- **领域范围**：涵盖农学、医学、工程学、人文学科、自然科学等共261个学科

## 核心特性

- 包含899道测试题，覆盖261个细粒度学科
- 每道题在最多10个选项（A–J）中仅有一个正确答案
- 提供每个选项的解释（用于训练/分析，不对模型展示）
- 旨在测试深层领域知识，而非检索能力或常识推理
- 在2077AI首次提出，强调学科代表性

## 评估说明

- 默认使用 **test** 划分进行评估（899个样本）
- 主要指标：**准确率**（acc）—— 单次推理模式下的 Pass@1
- 采用0-shot思维链（CoT）评估方式，从 ``ANSWER: [LETTER]`` 标记中提取答案
- 每个样本均附带学科元数据，可在评估结果中查看；但未按学科划分子集
- [GitHub](https://github.com/weihao1115/KINA-Benchmark)


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `kina` |
| **数据集ID** | [evalscope/KINA](https://modelscope.cn/datasets/evalscope/KINA/summary) |
| **论文** | [Paper](https://www.2077ai.com/kina) |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 899 |
| 提示词长度（平均） | 3280.88 字符 |
| 提示词长度（最小/最大） | 482 / 22536 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "4750dae8",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J. Think step by step before answering.\n\nUnder con ... [TRUNCATED 1998 chars] ...  it more economically sustainable under concentrate-restricted conditions.\nJ) Choose barn-dried hay because its intact fiber structure significantly increases milk fat percentage, making it more suitable for producing high-fat dairy products."
    }
  ],
  "choices": [
    "Silage, because anaerobic fermentation preserves soluble carbohydrates, true protein, and vitamins effectively, resulting in higher metabolizable energy density, superior palatability, and greater dry matter intake(DMI), thereby helping sustain milk yield when dietary concentrate is limited.",
    "Barn-dried hay, as it promotes higher DMI, enabling adequate nutrient intake and improving nitrogen utilization efficiency despite its lower crude protein concentration.",
    "Both forages are functionally equivalent and can be substituted on an equal dry matter basis, as they are both classified as roughages and exert no significant differential effect on lactation performance.",
    "Barn-dried hay, owing to its physically effective fiber structure and high lignin content, which enhance rumination activity and mitigate the risk of subacute ruminal acidosis.",
    "Silage, due to its high moisture content, which reduces voluntary water consumption and contributes to on-farm water conservation.",
    "Choose barn-dried hay because it contains no moisture, has a high dry matter content, and is therefore more \"nutrient-concentrated\" than wet silage.",
    "Choose silage because it contains probiotics that can directly improve gut health in dairy cows and serve as a protein source to replace concentrate.",
    "Choose silage. In southern China's rainy climate, hay is prone to mold growth and aflatoxin contamination, whereas silage avoids this risk and ensures raw milk safety-particularly important when concentrate supply is limited and reliance on safe forage is critical.",
    "Choose silage because it can be produced locally(e.g., whole-plant corn), harvested and stored mechanically, and offers lower cost per unit of nutrient compared to purchased high-quality hay, making it more economically sustainable under concentrate-restricted conditions.",
    "Choose barn-dried hay because its intact fiber structure significantly increases milk fat percentage, making it more suitable for producing high-fat dairy products."
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "index": 0,
    "discipline": "Agronomy/Animal Husbandry/Animal Nutrition and Feed Science"
  }
}
```

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
    --datasets kina \
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
    datasets=['kina'],
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```
