# MMLU-Redux


## 概述

MMLU-Redux 是 MMLU 基准测试的一个改进版本，修正了原始数据集中的错误答案。它通过修复不正确的标准答案标签、缺失的正确选项以及存在歧义的问题，解决了原始 MMLU 数据集中已知的错误。

## 任务描述

- **任务类型**：多项选择知识评估
- **输入**：包含四个选项的问题
- **输出**：正确答案的字母（A/B/C/D）
- **领域**：涵盖 STEM、人文学科、社会科学及其他领域的 57 个学科

## 主要特性

- 修正了原始 MMLU 基准测试中的错误
- 修复的错误类型包括：
  - `no_correct_answer`：缺少正确选项的问题
  - `wrong_groundtruth`：标准答案错误的问题
  - `multiple_correct_answers`：答案存在歧义的问题
- 与原始 MMLU 保持相同的 57 个学科覆盖范围
- 与 MMLU 评估框架兼容

## 评估说明

- 默认使用 **test** 数据划分进行评估
- 主要指标：**准确率（Accuracy）**（对多答案问题使用包含标志）
- 使用思维链（Chain-of-Thought, CoT）提示
- 仅支持零样本（zero-shot）评估（不支持少样本）
- 结果按学科和类别（STEM、人文学科、社会科学、其他）汇总


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mmlu_redux` |
| **数据集 ID** | [AI-ModelScope/mmlu-redux-2.0](https://modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认样本数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 5,700 |
| 提示词长度（平均） | 600.81 字符 |
| 提示词长度（最小/最大） | 255 / 5082 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `abstract_algebra` | 100 | 399.13 | 285 | 525 |
| `anatomy` | 100 | 454.09 | 323 | 788 |
| `astronomy` | 100 | 516.53 | 297 | 1087 |
| `business_ethics` | 100 | 538.35 | 274 | 927 |
| `clinical_knowledge` | 100 | 453.01 | 303 | 757 |
| `college_biology` | 100 | 550.59 | 343 | 1081 |
| `college_chemistry` | 100 | 451.35 | 292 | 890 |
| `college_computer_science` | 100 | 631.74 | 336 | 1344 |
| `college_mathematics` | 100 | 451.77 | 255 | 709 |
| `college_medicine` | 100 | 669.42 | 308 | 5082 |
| `college_physics` | 100 | 500.71 | 317 | 797 |
| `computer_security` | 100 | 476.01 | 291 | 1115 |
| `conceptual_physics` | 100 | 372.72 | 284 | 539 |
| `econometrics` | 100 | 610.01 | 304 | 1036 |
| `electrical_engineering` | 100 | 373.01 | 286 | 585 |
| `elementary_mathematics` | 100 | 403.98 | 264 | 797 |
| `formal_logic` | 100 | 598.04 | 318 | 1275 |
| `global_facts` | 100 | 389.82 | 303 | 746 |
| `high_school_biology` | 100 | 574.54 | 328 | 1078 |
| `high_school_chemistry` | 100 | 496.56 | 271 | 1093 |
| `high_school_computer_science` | 100 | 649.34 | 278 | 1786 |
| `high_school_european_history` | 100 | 1840.88 | 855 | 3045 |
| `high_school_geography` | 100 | 417.76 | 306 | 616 |
| `high_school_government_and_politics` | 100 | 539.42 | 384 | 864 |
| `high_school_macroeconomics` | 100 | 509.35 | 319 | 756 |
| `high_school_mathematics` | 100 | 409.45 | 282 | 846 |
| `high_school_microeconomics` | 100 | 508.99 | 325 | 1070 |
| `high_school_physics` | 100 | 600.82 | 324 | 1414 |
| `high_school_psychology` | 100 | 476.49 | 309 | 1055 |
| `high_school_statistics` | 100 | 736.04 | 376 | 1772 |
| `high_school_us_history` | 100 | 1643.04 | 782 | 2595 |
| `high_school_world_history` | 100 | 1749.59 | 749 | 3834 |
| `human_aging` | 100 | 416.87 | 323 | 689 |
| `human_sexuality` | 100 | 457.5 | 307 | 1145 |
| `international_law` | 100 | 636.61 | 332 | 986 |
| `jurisprudence` | 100 | 518.24 | 307 | 1039 |
| `logical_fallacies` | 100 | 516.62 | 333 | 902 |
| `machine_learning` | 100 | 513.09 | 315 | 806 |
| `management` | 100 | 399.38 | 294 | 645 |
| `marketing` | 100 | 474.85 | 335 | 757 |
| `medical_genetics` | 100 | 414.32 | 292 | 658 |
| `miscellaneous` | 100 | 388.56 | 277 | 1268 |
| `moral_disputes` | 100 | 534.28 | 325 | 872 |
| `moral_scenarios` | 100 | 620.62 | 563 | 767 |
| `nutrition` | 100 | 505.6 | 318 | 926 |
| `philosophy` | 100 | 462.44 | 319 | 1155 |
| `prehistory` | 100 | 468.62 | 320 | 943 |
| `professional_accounting` | 100 | 650.46 | 341 | 1226 |
| `professional_law` | 100 | 1370.21 | 359 | 2928 |
| `professional_medicine` | 100 | 1003.93 | 610 | 1735 |
| `professional_psychology` | 100 | 577.98 | 317 | 1502 |
| `public_relations` | 100 | 472.39 | 300 | 1188 |
| `security_studies` | 100 | 1029.35 | 317 | 2066 |
| `sociology` | 100 | 530.23 | 335 | 834 |
| `us_foreign_policy` | 100 | 490.28 | 305 | 754 |
| `virology` | 100 | 447.81 | 302 | 1383 |
| `world_religions` | 100 | 353.31 | 287 | 557 |

## 样例示例

**子集**: `abstract_algebra`

```json
{
  "input": [
    {
      "id": "f937b8b4",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\nStatement 1 | If T: V -> W is a linear transformation and dim(V ) < dim(W) < 1, then T must be injective. Statement 2 | Let dim(V) = n and suppose that T: V -> V is linear. If T is injective, then it is a bijection.\n\nA) True, True\nB) False, False\nC) True, False\nD) False, True"
    }
  ],
  "choices": [
    "True, True",
    "False, False",
    "True, False",
    "False, True"
  ],
  "target": [
    "A"
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "error_type": "bad_question_clarity",
    "correct_answer": "0",
    "potential_reason": "Statement 2 is true and well defined. \r\nHowever, statement 1 is not well defined: The dimension of a vector space is a nonnegative number, and since dim(V) < dim(W) < 1, this means dim(V) has to be negative. Taking this statement literally, the implication is vacuously true as the premise cannot be satisfed, but I doubt that was what the question is trying to test. "
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

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mmlu_redux \
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
    datasets=['mmlu_redux'],
    dataset_args={
        'mmlu_redux': {
            # subset_list: ['abstract_algebra', 'anatomy', 'astronomy']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```