# MusicTrivia

## Overview

MusicTrivia is a curated multiple-choice benchmark for evaluating AI models on music knowledge. It covers both classical and modern music topics including composers, musical periods, instruments, and popular artists.

## Task Description

- **Task Type**: Multiple-Choice Question Answering
- **Input**: Music-related trivia question with multiple choice options
- **Output**: Selected correct answer
- **Domains**: Classical music, modern music, music history

## Key Features

- Comprehensive coverage of music domains
- Questions about composers, periods, and artists
- Tests factual recall and domain knowledge
- Curated for quality and accuracy
- Balanced difficulty across topics

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Primary metric: **Accuracy**
- Evaluates on **test** split
- Uses standard single-answer multiple-choice template

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `music_trivia` |
| **Dataset ID** | [extraordinarylab/music-trivia](https://modelscope.cn/datasets/extraordinarylab/music-trivia/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 512 |
| Prompt Length (Mean) | 412.5 chars |
| Prompt Length (Min/Max) | 256 / 1005 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "d16392be",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nBeethoven's third period was distinct from his second on account of all the following factors EXCEPT\n\nA) The influence of Italian opera on his compositions\nB) The occupation of Vienna by French troops\nC) The introduction of new instruments in the orchestra\nD) The emergence of Romanticism in music"
    }
  ],
  "choices": [
    "The influence of Italian opera on his compositions",
    "The occupation of Vienna by French troops",
    "The introduction of new instruments in the orchestra",
    "The emergence of Romanticism in music"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets music_trivia \
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
    datasets=['music_trivia'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


