# ScienceQA


## Overview

ScienceQA is a multimodal benchmark consisting of multiple-choice science questions derived from elementary and high school curricula. It covers diverse subjects including natural science, social science, and language science, with questions accompanied by both image and text contexts.

## Task Description

- **Task Type**: Multimodal Science Question Answering
- **Input**: Question with optional image context + multiple choices
- **Output**: Correct answer choice letter
- **Domains**: Natural science, social science, language science

## Key Features

- Questions sourced from real K-12 science curricula
- Most questions include both image and text contexts
- Annotated with detailed lectures and explanations
- Supports research into chain-of-thought reasoning
- Covers multiple grade levels and difficulty ranges
- Rich metadata including topic, skill, and category information

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** on multiple-choice questions
- Uses Chain-of-Thought (CoT) prompting for reasoning
- Metadata includes solution explanations for analysis
- Questions span grades from elementary to high school


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `science_qa` |
| **Dataset ID** | [AI-ModelScope/ScienceQA](https://modelscope.cn/datasets/AI-ModelScope/ScienceQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiModal` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,241 |
| Prompt Length (Mean) | 370.49 chars |
| Prompt Length (Min/Max) | 250 / 1037 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 2,017 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 170x77 - 750x625 |
| Formats | png |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "7586b8fe",
      "content": [
        {
          "text": "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B. Think step by step before answering.\n\nWhich figure of speech is used in this text?\nSing, O goddess, the anger of Achilles son of Peleus, that brought countless ills upon the Achaeans.\n—Homer, The Iliad\n\nA) chiasmus\nB) apostrophe"
        }
      ]
    }
  ],
  "choices": [
    "chiasmus",
    "apostrophe"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "hint": "",
    "task": "closed choice",
    "grade": "grade11",
    "subject": "language science",
    "topic": "figurative-language",
    "category": "Literary devices",
    "skill": "Classify the figure of speech: anaphora, antithesis, apostrophe, assonance, chiasmus, understatement",
    "lecture": "Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\nAnaphora is the repetition of the same word or words at the beginning of several phrases or clauses.\nWe are united ... [TRUNCATED] ... but reverses the order of words.\nNever let a fool kiss you or a kiss fool you.\nUnderstatement involves deliberately representing something as less serious or important than it really is.\nAs you know, it can get a little cold in the Antarctic.",
    "solution": "The text uses apostrophe, a direct address to an absent person or a nonhuman entity.\nO goddess is a direct address to a goddess, a nonhuman entity."
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
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
    --datasets science_qa \
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
    datasets=['science_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


