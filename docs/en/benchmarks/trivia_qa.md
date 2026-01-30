# TriviaQA


## Overview

TriviaQA is a large-scale reading comprehension dataset containing over 650K question-answer-evidence triples. Questions are collected from trivia enthusiast websites and paired with Wikipedia articles as evidence documents.

## Task Description

- **Task Type**: Reading Comprehension / Question Answering
- **Input**: Question with Wikipedia context passage
- **Output**: Answer extracted or generated from context
- **Domain**: General knowledge trivia questions

## Key Features

- 650K+ question-answer-evidence triples
- Questions written by trivia enthusiasts (naturally challenging)
- Multiple valid answer aliases for flexible evaluation
- Wikipedia articles provide evidence passages
- Tests both reading comprehension and knowledge retrieval

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses the Wikipedia reading comprehension subset (rc.wikipedia)
- Answers should follow the format: "ANSWER: [ANSWER]"
- Supports inclusion-based matching for answer comparison
- Evaluates on validation split


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `trivia_qa` |
| **Dataset ID** | [evalscope/trivia_qa](https://modelscope.cn/datasets/evalscope/trivia_qa/summary) |
| **Paper** | N/A |
| **Tags** | `QA`, `ReadingComprehension` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `validation` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 7,993 |
| Prompt Length (Mean) | 54126.56 chars |
| Prompt Length (Min/Max) | 339 / 691325 chars |

## Sample Example

**Subset**: `rc.wikipedia`

```json
{
  "input": [
    {
      "id": "545e4eda",
      "content": "Read the content and answer the following question.\n\nContent: ['Andrew Lloyd Webber, Baron Lloyd-Webber   (born 22 March 1948) is an English composer and impresario of musical theatre. \\n\\nSeveral of his musicals have run for more than a deca ... [TRUNCATED] ... ening titles.']\n\nQuestion: Which Lloyd Webber musical premiered in the US on 10th December 1993?\n\nKeep your The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem.\n"
    }
  ],
  "target": [
    "Sunset Blvd",
    "West Sunset Boulevard",
    "Sunset Boulevard",
    "Sunset Bulevard",
    "Sunset Blvd.",
    "sunset boulevard",
    "sunset bulevard",
    "west sunset boulevard",
    "sunset blvd"
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "tc_33",
    "content": [
      "Andrew Lloyd Webber, Baron Lloyd-Webber   (born 22 March 1948) is an English composer and impresario of musical theatre. \n\nSeveral of his musicals have run for more than a decade both in the West End and on Broadway. He has composed 13 musica ... [TRUNCATED] ... same name, composed the song \"Fields of Sun\". The actual song was never used on the show, nor was it available on the CD soundtrack that was released at the time. He was however still credited for the unused song in the show's opening titles."
    ]
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Read the content and answer the following question.

Content: {content}

Question: {question}

Keep your The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets trivia_qa \
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
    datasets=['trivia_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


