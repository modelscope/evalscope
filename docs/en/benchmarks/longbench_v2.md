# LongBench-v2


## Overview

LongBench v2 is a challenging benchmark for evaluating long-context understanding of large language models. It covers a wide variety of real-world tasks that require reading and comprehending long documents (ranging from a few thousand to over 2 million tokens), spanning multiple domains such as single-document QA, multi-document QA, long in-context learning, long-structured data understanding, and code repository understanding.

## Task Description

- **Task Type**: Long-Context Multiple-Choice Question Answering
- **Input**: Long document context + multiple-choice question with four answer choices (A, B, C, D)
- **Output**: Single correct answer letter
- **Domains**: Single-Doc QA, Multi-Doc QA, Long In-Context Learning, Long Structured Data Understanding, Code Repo Understanding
- **Difficulty**: Easy / Hard
- **Length**: Short / Medium / Long

## Key Features

- 503 high-quality questions requiring genuine long-document understanding
- Context lengths ranging from a few thousand tokens to over 2 million tokens
- Questions are bilingual (English and Chinese)
- Designed to require careful reading; correct answers cannot be guessed without reading the document
- Covers diverse real-world application scenarios

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (train split used as test set)
- Primary metric: **Accuracy** (exact match on letter choice)
- All four answer choices are required; no random shuffling needed
- Samples are split into **3 subsets by context length**: `short`, `medium`, `long`
- Use `subset_list` to evaluate specific length subsets (e.g., `['short', 'medium']`)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `longbench_v2` |
| **Dataset ID** | [ZhipuAI/LongBench-v2](https://modelscope.cn/datasets/ZhipuAI/LongBench-v2/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `MCQ`, `ReadingComprehension` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 503 |
| Prompt Length (Mean) | 872928.83 chars |
| Prompt Length (Min/Max) | 49433 / 16184015 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `short` | 180 | 124200.42 | 49433 | 841252 |
| `medium` | 215 | 501002.72 | 172108 | 2233351 |
| `long` | 108 | 2861217.94 | 720823 | 16184015 |

## Sample Example

**Subset**: `short`

```json
{
  "input": [
    {
      "id": "7e9a926f",
      "content": "Please read the following text and answer the questions below.\n\n<text>\nContents\nPreface.\n................................................................................................ 67\nI. China’s Court System and Reform Process.\n......... ... [TRUNCATED 163697 chars] ...  accelerate the construction of intelligent courts.\nC) Improve the work ability of office staff and strengthen the reserve of work knowledge.\nD) Use advanced information systems to improve the level of information technology in case handling."
    }
  ],
  "choices": [
    "Through technology empowerment, change the way of working and improve office efficiency.",
    "Establish new types of courts, such as intellectual property courts, financial courts, and Internet courts, and accelerate the construction of intelligent courts.",
    "Improve the work ability of office staff and strengthen the reserve of work knowledge.",
    "Use advanced information systems to improve the level of information technology in case handling."
  ],
  "target": "D",
  "id": 0,
  "group_id": 0,
  "subset_key": "short",
  "metadata": {
    "domain": "Single-Document QA",
    "sub_domain": "Financial",
    "difficulty": "easy",
    "length": "short",
    "context": "Contents\nPreface.\n................................................................................................ 67\nI. China’s Court System and Reform Process.\n.................................... 68\nII. Fully Implementing the Judicial Acco ... [TRUNCATED 162872 chars] ... a better environment for socialist rule of \nlaw, advance the judicial civilization to a higher level, and strive to make the \npeople obtain fair and just outcomes in every judicial case.\n法院的司法改革（2013-2018）.indd   161\n2019/03/01,星期五   17:42:05",
    "_id": "66f36490821e116aacb2cc22"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Please read the following text and answer the questions below.

<text>
{document}
</text>

Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets longbench_v2 \
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
    datasets=['longbench_v2'],
    dataset_args={
        'longbench_v2': {
            # subset_list: ['short', 'medium', 'long']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


