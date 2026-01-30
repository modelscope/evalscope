# CMMMU


## Overview

CMMU (Chinese Massive Multi-discipline Multimodal Understanding) includes manually collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines in Chinese. It is the Chinese counterpart to MMMU.

## Task Description

- **Task Type**: Chinese Multimodal Question Answering
- **Input**: Image(s) + question in Chinese with answer choices
- **Output**: Correct answer choice
- **Language**: Chinese

## Key Features

- 30 subjects across 6 core disciplines
- Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering
- 39 heterogeneous image types (charts, diagrams, maps, tables, etc.)
- College-level difficulty
- Multiple question types (multiple-choice, true/false, short answer)

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on validation split
- Simple accuracy metric
- Chinese language prompts used


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cmmmu` |
| **Dataset ID** | [lmms-lab/CMMMU](https://modelscope.cn/datasets/lmms-lab/CMMMU/summary) |
| **Paper** | N/A |
| **Tags** | `Chinese`, `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `val` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 900 |
| Prompt Length (Mean) | 185.59 chars |
| Prompt Length (Min/Max) | 91 / 1045 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `设计` | 18 | 216.39 | 136 | 347 |
| `音乐` | 21 | 149.57 | 110 | 185 |
| `艺术` | 16 | 166.44 | 143 | 240 |
| `艺术理论` | 33 | 171.21 | 118 | 268 |
| `经济` | 20 | 227.3 | 105 | 412 |
| `会计` | 39 | 197.05 | 103 | 528 |
| `金融` | 29 | 166 | 106 | 333 |
| `管理` | 22 | 236.73 | 135 | 423 |
| `营销` | 16 | 191.62 | 131 | 370 |
| `物理` | 35 | 266.11 | 146 | 497 |
| `地理` | 49 | 160.02 | 93 | 326 |
| `化学` | 42 | 187.4 | 103 | 322 |
| `生物` | 35 | 162.49 | 91 | 293 |
| `数学` | 43 | 188.23 | 101 | 370 |
| `临床医学` | 28 | 183.39 | 94 | 310 |
| `公共卫生` | 46 | 183.04 | 110 | 271 |
| `基础医学` | 32 | 146.56 | 92 | 219 |
| `诊断学与实验室医学` | 12 | 150.08 | 116 | 189 |
| `制药` | 35 | 181.77 | 97 | 413 |
| `历史` | 25 | 185.84 | 111 | 264 |
| `心理学` | 29 | 144.83 | 96 | 207 |
| `文献学` | 7 | 163.86 | 109 | 227 |
| `社会学` | 24 | 165 | 100 | 249 |
| `计算机科学` | 35 | 218.89 | 101 | 532 |
| `电子学` | 29 | 179.48 | 110 | 318 |
| `机械工程` | 40 | 196.1 | 98 | 817 |
| `能源和电力` | 32 | 186.44 | 100 | 330 |
| `材料` | 38 | 194.5 | 102 | 476 |
| `建筑学` | 49 | 175.39 | 97 | 415 |
| `农业` | 21 | 215.86 | 96 | 1045 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,023 |
| Images per Sample | min: 1, max: 5, mean: 1.14 |
| Resolution Range | 112x38 - 1500x3000 |
| Formats | jpeg, png |


## Sample Example

**Subset**: `设计`

```json
{
  "input": [
    {
      "id": "47c0e169",
      "content": [
        {
          "text": "请回答以下多项选择题，并选出正确选项。这些题目可能包括单选和多选题型。如果所提供的信息不足以确定一个明确的答案，那么请根据可用的数据和你的判断来选择最可能正确的选项。\n\n问题："
        },
        {
          "image": "[BASE64_IMAGE: png, ~17.4KB]"
        },
        {
          "text": "为一幅灰度图，要为它局部添加颜色以得到右图所示的效果，正确的操作步骤是（ ）。\n选项：\n(A) 先将色彩模式转为RGB，然后用工具箱中的 【画笔工具】上色\n(B) 先将色彩模式转为RGB，制作局部选区，然后打开【色相/饱和度】对话框，在其中点中【着色】项,调节色彩属性参数\n(C) 先将色彩模式转为RGB，制作局部选区，然后打开【可选颜色】对话框,调节参数\n(D) 打开【色相/饱和度】对话框，直接调节色彩属性参数\n\n正确答案：\n"
        }
      ]
    }
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "设计",
  "metadata": {
    "id": "1900",
    "type": "选择",
    "source_type": "website",
    "analysis": null,
    "distribution": "本科",
    "difficulty_level": "easy",
    "subcategory": "设计",
    "category": "艺术与设计",
    "subfield": "['图像编辑', '色彩调整']",
    "img_type": "['屏幕截图']",
    "answer": "B",
    "option1": "先将色彩模式转为RGB，然后用工具箱中的 【画笔工具】上色",
    "option2": "先将色彩模式转为RGB，制作局部选区，然后打开【色相/饱和度】对话框，在其中点中【着色】项,调节色彩属性参数",
    "option3": "先将色彩模式转为RGB，制作局部选区，然后打开【可选颜色】对话框,调节参数",
    "option4": "打开【色相/饱和度】对话框，直接调节色彩属性参数"
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cmmmu \
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
    datasets=['cmmmu'],
    dataset_args={
        'cmmmu': {
            # subset_list: ['设计', '音乐', '艺术']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


