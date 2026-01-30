# IQuiz


## Overview

IQuiz is a Chinese benchmark for evaluating AI models on intelligence quotient (IQ) and emotional quotient (EQ) questions. It tests logical reasoning, pattern recognition, and social-emotional understanding through multiple-choice questions.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Question in Chinese with multiple choice options
- **Output**: Selected answer with explanation (Chain-of-Thought)
- **Language**: Chinese

## Key Features

- Dual evaluation of IQ and EQ capabilities
- Chinese-language cognitive assessment
- Multiple difficulty levels
- Requires explanation alongside answer selection
- Tests logical reasoning and emotional understanding

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- Subsets: **IQ** (logical reasoning) and **EQ** (emotional intelligence)
- Uses Chinese Chain-of-Thought prompt template
- Evaluates on **test** split
- Metadata includes difficulty level information


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `iquiz` |
| **Dataset ID** | [AI-ModelScope/IQuiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary) |
| **Paper** | N/A |
| **Tags** | `Chinese`, `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 120 |
| Prompt Length (Mean) | 248.31 chars |
| Prompt Length (Min/Max) | 146 / 394 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `IQ` | 40 | 194.5 | 146 | 323 |
| `EQ` | 80 | 275.21 | 219 | 394 |

## Sample Example

**Subset**: `IQ`

```json
{
  "input": [
    {
      "id": "748f2700",
      "content": "回答下面的单项选择题，请选出其中的正确答案。你的回答的最后一行应该是这样的格式：\"答案：[LETTER]\"（不带引号），其中 [LETTER] 是 A,B,C,D 中的一个。请在回答前进行一步步思考。\n\n问题：天气预报说本周星期三会下雨，昨天果然下雨了，今天星期几？\n选项：\nA) 星期一\nB) 星期二\nC) 星期三\nD) 星期四\n"
    }
  ],
  "choices": [
    "星期一",
    "星期二",
    "星期三",
    "星期四"
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "level": 1
  }
}
```

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
    --datasets iquiz \
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
    datasets=['iquiz'],
    dataset_args={
        'iquiz': {
            # subset_list: ['IQ', 'EQ']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


