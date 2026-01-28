# CMMU


## Overview

CMMU is a novel Chinese multi-modal benchmark designed to evaluate domain-specific knowledge across seven foundational subjects: math, biology, physics, chemistry, geography, politics, and history. It tests multimodal understanding in Chinese educational contexts.

## Task Description

- **Task Type**: Chinese Multimodal Educational QA
- **Input**: Image + question in Chinese
- **Output**: Answer (multiple-choice, multiple-response, or fill-in-blank)
- **Language**: Chinese

## Key Features

- Seven foundational subjects
- Multiple question types (multiple-choice, multiple-response, fill-in-blank)
- Chinese K-12 educational content
- Tests domain-specific visual reasoning
- Diverse question formats

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Evaluates on validation split
- Numeric accuracy metric
- Chain-of-thought prompting for reasoning


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `cmmu` |
| **Dataset ID** | [evalscope/CMMU](https://modelscope.cn/datasets/evalscope/CMMU/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `val` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,800 |
| Prompt Length (Mean) | 282.54 chars |
| Prompt Length (Min/Max) | 139 / 1404 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `biology` | 270 | 284.66 | 140 | 854 |
| `chemistry` | 265 | 343.22 | 143 | 1404 |
| `geography` | 257 | 252.8 | 149 | 697 |
| `history` | 174 | 226.12 | 139 | 545 |
| `math` | 387 | 277.11 | 149 | 714 |
| `physics` | 270 | 317.33 | 162 | 710 |
| `politics` | 177 | 245.96 | 154 | 740 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 1,800 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 121x20 - 2327x1809 |
| Formats | gif, jpeg, png |


## Sample Example

**Subset**: `biology`

```json
{
  "input": [
    {
      "id": "17325c48",
      "content": [
        {
          "text": "回答下面的多项选择题，请选出其中的所有正确答案。你的回答的最后一行应该是这样的格式：\"答案：[LETTERS]\"（不带引号），其中 [LETTERS] 是 A,B,C,D 中的一个或多个。请在回答前进行一步步思考。\n\n问题：如图是培育抗除草剂玉米的技术路线图，含有内含子的报告基因只能在真核生物中正确表达，其产物能催化无色物质K呈现蓝色。转化过程中愈伤组织表面常残留农杆菌，会导致未转化的愈伤组织可能在含除草剂的培养基中生长。下列相关叙述正确的是（　　）\n选项：\nA) 过程①用两种限制酶就可防止酶切产物自身环化\nB) 过程②用Ca2+处理可提高转化成功率\nC) 过程③应在培养基中加入除草剂和物质K\nD) 筛选得到的A是无农杆菌附着的转化愈伤组织\n"
        },
        {
          "image": "[BASE64_IMAGE: png, ~34.9KB]"
        }
      ]
    }
  ],
  "choices": [
    "过程①用两种限制酶就可防止酶切产物自身环化",
    "过程②用Ca2+处理可提高转化成功率",
    "过程③应在培养基中加入除草剂和物质K",
    "筛选得到的A是无农杆菌附着的转化愈伤组织"
  ],
  "target": "BC",
  "id": 0,
  "group_id": 0,
  "subset_key": "biology",
  "metadata": {
    "type": "multiple-response",
    "grade_band": "high",
    "difficulty": "hard",
    "split": "val",
    "subject": "biology",
    "sub_questions": null,
    "solution_info": "解：A、过程①要用两种特定的限制酶，切割出两个不同的黏性末端序列，从而防止酶切产物自身环化，A错误；B、农杆菌属于原核生物，所以过程②用Ca2+处理，使其成为感受态细胞，可提高转化成功率，B正确；C、根据题意可知，报告基因的产物能催化无色物质K呈现蓝色，而愈伤组织表面残留的农杆菌会导致未转化的愈伤组织能在含除草剂的培养基中生长，因此过程③应在培养基中加入除草剂和物质K，C正确；D、筛选得到的A是有农杆菌附着的转化愈伤组织，D错误。故选：BC。基因工程技术的基本步骤：（1）目的基因 ... [TRUNCATED] ... 入动物细胞最有效的方法是显微注射法；将目的基因导入微生物细胞的方法是感受态细胞法。（4）目的基因的检测与鉴定：分子水平上的检测：①检测转基因生物染色体的DNA是否插入目的基因--DNA分子杂交技术；②检测目的基因是否转录出了mRNA--分子杂交技术；③检测目的基因是否翻译成蛋白质--抗原-抗体杂交技术。个体水平上的鉴定：抗虫鉴定、抗病鉴定、活性鉴定等。本题考查基因工程的相关知识，要求考生识记基因工程的原理及操作步骤，掌握各操作步骤中需要注意的细节问题，能结合题图信息准确判断各项。",
    "id": "biology_166"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式："答案：[LETTER]"（不带引号），其中 [LETTER] 是 {letters} 中的一个。请在回答前进行一步步思考。

问题：{question}
选项：
{choices}

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets cmmu \
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
    datasets=['cmmu'],
    dataset_args={
        'cmmu': {
            # subset_list: ['biology', 'chemistry', 'geography']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


