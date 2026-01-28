# SuperGPQA


## 概述

SuperGPQA 是一个大规模多项选择题问答数据集，旨在评估模型在不同领域的泛化能力。该数据集包含来自 50 多个领域的 26,000 多道题目，每道题提供 10 个选项。

## 任务描述

- **任务类型**：多项选择知识评估
- **输入**：包含 10 个选项（A-J）的问题
- **输出**：正确答案的字母
- **领域**：涵盖科学、工程、医学、经济学、法学等 50 多个领域

## 主要特点

- 覆盖 50 多个学术领域的 26,000+ 道问题
- 每题提供 10 个选项（比标准的 4 选项更具挑战性）
- 广泛覆盖以下领域：
  - 科学：数学、物理、化学、生物
  - 工程：计算机科学、电气、机械
  - 医学：临床医学、基础医学、药学
  - 人文学科：哲学、历史、文学
  - 社会科学：经济学、法学、社会学

## 评估说明

- 默认使用 **train** 划分进行评估（唯一可用划分）
- 主要指标：多项选择题的 **准确率（Accuracy）**
- 仅支持 0-shot 或 5-shot 评估
- 使用思维链（Chain-of-Thought, CoT）提示
- 结果可按领域或学科类别分组统计

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `super_gpqa` |
| **数据集 ID** | [m-a-p/SuperGPQA](https://modelscope.cn/datasets/m-a-p/SuperGPQA/summary) |
| **论文** | N/A |
| **标签** | `Knowledge`, `MCQ` |
| **指标** | `acc` |
| **默认 Shots** | 0-shot |
| **评估划分** | `train` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 26,529 |
| 提示词长度（平均） | 826.04 字符 |
| 提示词长度（最小/最大） | 294 / 8444 字符 |

**各子集统计数据：**

| 子集 | 样本数 | 提示平均长度 | 提示最小长度 | 提示最大长度 |
|--------|---------|-------------|------------|------------|
| `Electronic Science and Technology` | 246 | 994.09 | 355 | 6149 |
| `Philosophy` | 347 | 880.01 | 357 | 3171 |
| `Traditional Chinese Medicine` | 268 | 691.78 | 339 | 2282 |
| `Applied Economics` | 723 | 779.14 | 348 | 5407 |
| `Mathematics` | 2,622 | 908.66 | 326 | 7946 |
| `Physics` | 2,845 | 936.31 | 348 | 6780 |
| `Clinical Medicine` | 1,218 | 830.98 | 334 | 4003 |
| `Computer Science and Technology` | 763 | 823.37 | 331 | 4409 |
| `Information and Communication Engineering` | 504 | 921.14 | 347 | 5594 |
| `Control Science and Engineering` | 190 | 1042.88 | 354 | 6016 |
| `Theoretical Economics` | 150 | 843.95 | 363 | 2432 |
| `Law` | 591 | 1220.98 | 366 | 4267 |
| `History` | 674 | 595.18 | 326 | 3707 |
| `Basic Medicine` | 567 | 690.19 | 320 | 2368 |
| `Education` | 247 | 695.45 | 355 | 1728 |
| `Materials Science and Engineering` | 289 | 834.37 | 359 | 2630 |
| `Electrical Engineering` | 556 | 891.89 | 385 | 4025 |
| `Systems Science` | 50 | 1214.34 | 403 | 2676 |
| `Power Engineering and Engineering Thermophysics` | 684 | 885.2 | 345 | 3093 |
| `Military Science` | 205 | 709.2 | 349 | 2839 |
| `Biology` | 1,120 | 786.49 | 337 | 4581 |
| `Business Administration` | 142 | 877.2 | 359 | 2219 |
| `Language and Literature` | 440 | 778.78 | 313 | 4331 |
| `Public Health and Preventive Medicine` | 292 | 670.73 | 348 | 3899 |
| `Political Science` | 65 | 700.06 | 362 | 2239 |
| `Chemistry` | 1,769 | 823.53 | 330 | 6109 |
| `Hydraulic Engineering` | 218 | 840.64 | 354 | 2504 |
| `Chemical Engineering and Technology` | 410 | 1035.03 | 325 | 3839 |
| `Pharmacy` | 278 | 642.21 | 355 | 2629 |
| `Geography` | 133 | 636.14 | 336 | 2514 |
| `Art Studies` | 603 | 565.14 | 338 | 3552 |
| `Architecture` | 162 | 647.03 | 345 | 2684 |
| `Forestry Engineering` | 100 | 707.91 | 366 | 2676 |
| `Public Administration` | 151 | 730.48 | 339 | 4242 |
| `Oceanography` | 200 | 801.77 | 332 | 2462 |
| `Journalism and Communication` | 207 | 550.58 | 341 | 1620 |
| `Nuclear Science and Technology` | 107 | 850.05 | 378 | 3467 |
| `Weapon Science and Technology` | 100 | 866.71 | 338 | 3090 |
| `Naval Architecture and Ocean Engineering` | 138 | 599.32 | 295 | 2320 |
| `Environmental Science and Engineering` | 189 | 779.83 | 333 | 2629 |
| `Transportation Engineering` | 251 | 778.94 | 362 | 3854 |
| `Geology` | 341 | 810.18 | 350 | 3293 |
| `Physical Oceanography` | 50 | 933.54 | 364 | 2176 |
| `Musicology` | 426 | 540.96 | 294 | 2909 |
| `Stomatology` | 132 | 792.47 | 386 | 2975 |
| `Aquaculture` | 56 | 497.89 | 339 | 1457 |
| `Mechanical Engineering` | 176 | 863.61 | 377 | 3684 |
| `Aeronautical and Astronautical Science and Technology` | 119 | 792.2 | 366 | 3178 |
| `Civil Engineering` | 358 | 831.25 | 343 | 2306 |
| `Mechanics` | 908 | 955.85 | 433 | 8444 |
| `Petroleum and Natural Gas Engineering` | 112 | 725.53 | 382 | 1882 |
| `Sociology` | 143 | 758.95 | 334 | 3241 |
| `Food Science and Engineering` | 109 | 600.39 | 343 | 1383 |
| `Agricultural Engineering` | 104 | 960.7 | 335 | 3178 |
| `Surveying and Mapping Science and Technology` | 168 | 729.06 | 340 | 2648 |
| `Metallurgical Engineering` | 255 | 828.58 | 373 | 2421 |
| `Library, Information and Archival Management` | 150 | 806.67 | 368 | 3502 |
| `Mining Engineering` | 100 | 865.04 | 379 | 5785 |
| `Astronomy` | 405 | 810.0 | 334 | 2925 |
| `Geological Resources and Geological Engineering` | 50 | 722 | 369 | 1554 |
| `Atmospheric Science` | 203 | 732.33 | 337 | 2455 |
| `Optical Engineering` | 376 | 777.42 | 366 | 2718 |
| `Animal Husbandry` | 103 | 621.8 | 361 | 1913 |
| `Geophysics` | 100 | 927.86 | 384 | 7376 |
| `Crop Science` | 145 | 662.8 | 394 | 1694 |
| `Management Science and Engineering` | 58 | 765.22 | 411 | 1832 |
| `Psychology` | 87 | 674.59 | 393 | 1973 |
| `Forestry` | 131 | 649.55 | 393 | 1995 |
| `Textile Science and Engineering` | 100 | 723.75 | 378 | 2124 |
| `Veterinary Medicine` | 50 | 602.5 | 362 | 1063 |
| `Instrument Science and Technology` | 50 | 759.36 | 375 | 1820 |
| `Physical Education` | 150 | 687.39 | 365 | 2156 |

## 样例示例

**子集**: `Electronic Science and Technology`

```json
{
  "input": [
    {
      "id": "d0ceb797",
      "content": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D,E,F,G,H,I,J. Think step by step before answering.\n\nThe commo ... [TRUNCATED] ... oduct of A1 and A2's common-mode rejection ratios\nG) the size of A2's common-mode rejection ratio\nH) the size of A1's common-mode rejection ratio\nI) The difference in the common-mode rejection ratio of A1 and A2 themselves\nJ) input resistance"
    }
  ],
  "choices": [
    "the absolute value of the difference in the common-mode rejection ratio of A1 and A2 themselves",
    "all of the above",
    "the average of A1 and A2's common-mode rejection ratios",
    "the sum of A1 and A2's common-mode rejection ratios",
    "the product of A1 and A2's common-mode rejection ratios",
    "the square root of the product of A1 and A2's common-mode rejection ratios",
    "the size of A2's common-mode rejection ratio",
    "the size of A1's common-mode rejection ratio",
    "The difference in the common-mode rejection ratio of A1 and A2 themselves",
    "input resistance"
  ],
  "target": "I",
  "id": 0,
  "group_id": 0,
  "subset_key": "Electronic Science and Technology",
  "metadata": {
    "field": "Electronic Science and Technology",
    "discipline": "Engineering",
    "uuid": "a8390c754538493ba59055689b4482aa",
    "explanation": "The difference in the common-mode rejection ratio of A1 and A2 themselves"
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

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets super_gpqa \
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
    datasets=['super_gpqa'],
    dataset_args={
        'super_gpqa': {
            # subset_list: ['Electronic Science and Technology', 'Philosophy', 'Traditional Chinese Medicine']  # 可选，用于评估特定子集
        }
    },
    limit=10,  # 正式评估时请删除此行
)

run_task(task_cfg=task_cfg)
```