# C-MMLU


## 概述

C-MMLU（Chinese Massive Multitask Language Understanding，中文大规模多任务语言理解）是一个全面的中文评估基准，涵盖 STEM、人文、社会科学以及中国特有主题等 67 个学科领域，用于评估模型在中文语境下的知识与推理能力。

## 任务描述

- **任务类型**：多项选择题问答（中文）
- **输入**：一道包含四个选项（A、B、C、D）的中文问题
- **输出**：单个正确答案的字母
- **学科范围**：67 个学科，按类别组织，包括中国特有主题

## 主要特点

- 覆盖 67 个多样化的中文知识领域
- 包含中国特有主题（如中国历史、文学、公务员考试等）
- 题目难度从基础教育到专业水平不等
- 同时考察通用知识与中国特有的文化知识
- 是中文语言模型评估的标准基准之一

## 评估说明

- 默认配置使用 **0-shot** 评估方式
- 使用中文思维链（Chain-of-Thought, CoT）提示模板
- 结果可按学科或类别进行聚合
- 类别包括：STEM、人文、社会科学、中国特有、其他
- 在测试集（test split）上进行评估

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `cmmlu` |
| **数据集 ID** | [evalscope/cmmlu](https://modelscope.cn/datasets/evalscope/cmmlu/summary) |
| **论文** | N/A |
| **标签** | `Chinese`, `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认示例数** | 0-shot |
| **评估分割** | `test` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 11,582 |
| 提示词长度（平均） | 197.87 字符 |
| 提示词长度（最小/最大） | 134 / 999 字符 |

**各子集统计信息：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `agronomy` | 169 | 168.76 | 142 | 266 |
| `anatomy` | 148 | 157.72 | 141 | 224 |
| `ancient_chinese` | 164 | 178.6 | 144 | 367 |
| `arts` | 160 | 161.22 | 141 | 233 |
| `astronomy` | 165 | 191.31 | 143 | 404 |
| `business_ethics` | 209 | 175.15 | 146 | 291 |
| `chinese_civil_service_exam` | 160 | 284.88 | 143 | 554 |
| `chinese_driving_rule` | 131 | 181.57 | 151 | 250 |
| `chinese_food_culture` | 136 | 170.49 | 139 | 270 |
| `chinese_foreign_policy` | 107 | 254.16 | 150 | 381 |
| `chinese_history` | 323 | 250.97 | 164 | 387 |
| `chinese_literature` | 204 | 177.55 | 145 | 397 |
| `chinese_teacher_qualification` | 179 | 207.4 | 156 | 326 |
| `college_actuarial_science` | 106 | 270.99 | 163 | 558 |
| `college_education` | 107 | 198.09 | 149 | 355 |
| `college_engineering_hydrology` | 106 | 189.62 | 146 | 273 |
| `college_law` | 108 | 220.14 | 157 | 310 |
| `college_mathematics` | 105 | 343.1 | 174 | 999 |
| `college_medical_statistics` | 106 | 212.85 | 151 | 450 |
| `clinical_knowledge` | 237 | 245.74 | 150 | 393 |
| `college_medicine` | 273 | 187.64 | 141 | 416 |
| `computer_science` | 204 | 187.76 | 143 | 516 |
| `computer_security` | 171 | 214.01 | 149 | 399 |
| `conceptual_physics` | 147 | 222.4 | 154 | 337 |
| `construction_project_management` | 139 | 186.58 | 149 | 306 |
| `economics` | 159 | 184.19 | 149 | 259 |
| `education` | 163 | 169.17 | 145 | 225 |
| `elementary_chinese` | 252 | 174.84 | 142 | 368 |
| `elementary_commonsense` | 198 | 163.93 | 139 | 247 |
| `elementary_information_and_technology` | 238 | 181.63 | 143 | 275 |
| `electrical_engineering` | 172 | 183.77 | 148 | 358 |
| `elementary_mathematics` | 230 | 184.92 | 145 | 320 |
| `ethnology` | 135 | 176.41 | 145 | 294 |
| `food_science` | 143 | 165.87 | 141 | 240 |
| `genetics` | 176 | 187.56 | 146 | 283 |
| `global_facts` | 149 | 182.32 | 146 | 329 |
| `high_school_biology` | 169 | 267.46 | 177 | 486 |
| `high_school_chemistry` | 132 | 260.74 | 160 | 395 |
| `high_school_geography` | 118 | 207.08 | 142 | 377 |
| `high_school_mathematics` | 164 | 203.72 | 151 | 356 |
| `high_school_physics` | 110 | 223.11 | 152 | 353 |
| `high_school_politics` | 143 | 269.18 | 174 | 386 |
| `human_sexuality` | 126 | 175.63 | 139 | 261 |
| `international_law` | 185 | 199.09 | 150 | 385 |
| `journalism` | 172 | 172.25 | 142 | 234 |
| `jurisprudence` | 411 | 226.57 | 146 | 514 |
| `legal_and_moral_basis` | 214 | 205.67 | 154 | 317 |
| `logical` | 123 | 181.72 | 143 | 427 |
| `machine_learning` | 122 | 213.32 | 155 | 419 |
| `management` | 210 | 180.32 | 145 | 287 |
| `marketing` | 180 | 185.59 | 144 | 247 |
| `marxist_theory` | 189 | 190.72 | 145 | 273 |
| `modern_chinese` | 116 | 207.66 | 142 | 471 |
| `nutrition` | 145 | 173.48 | 144 | 267 |
| `philosophy` | 105 | 179.91 | 143 | 359 |
| `professional_accounting` | 175 | 183.38 | 147 | 281 |
| `professional_law` | 211 | 231.1 | 150 | 414 |
| `professional_medicine` | 376 | 174.87 | 144 | 319 |
| `professional_psychology` | 232 | 173.55 | 142 | 273 |
| `public_relations` | 174 | 178.06 | 144 | 263 |
| `security_study` | 135 | 186.07 | 145 | 302 |
| `sociology` | 226 | 173.89 | 145 | 384 |
| `sports_science` | 165 | 170.49 | 141 | 283 |
| `traditional_chinese_medicine` | 185 | 165.38 | 134 | 240 |
| `virology` | 169 | 176.32 | 144 | 266 |
| `world_history` | 161 | 258.64 | 167 | 388 |
| `world_religions` | 160 | 163.08 | 142 | 235 |

## 样例示例

**子集**: `agronomy`

```json
{
  "input": [
    {
      "id": "4e04de48",
      "content": "回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：\"答案：[LETTER]\"（不带引号），其中 [LETTER] 是 A,B,C,D 中的一个。请在回答前进行一步步思考。\n\n问题：在农业生产中被当作极其重要的劳动对象发挥作用，最主要的不可替代的基本生产资料是\n选项：\nA) 农业生产工具\nB) 土地\nC) 劳动力\nD) 资金\n"
    }
  ],
  "choices": [
    "农业生产工具",
    "土地",
    "劳动力",
    "资金"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "agronomy",
  "metadata": {
    "subject": "agronomy"
  }
}
```

## 提示模板

**提示模板：**
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cmmlu \
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
    datasets=['cmmlu'],
    dataset_args={
        'cmmlu': {
            # subset_list: ['agronomy', 'anatomy', 'ancient_chinese']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```