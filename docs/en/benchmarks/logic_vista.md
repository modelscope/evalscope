# LogicVista


## Overview

LogicVista evaluates the fundamental logical reasoning abilities of multimodal large language models in visual contexts. Every item is a multiple-choice question whose answer options are drawn inside the image (diagrams, puzzles, sequences, charts), so a model must read the visual options and reason over them rather than over textual choices.

## Task Description

- **Task Type**: Visual Logical Reasoning (Multiple Choice)
- **Input**: Image containing the labelled answer options + question text
- **Output**: The label(s) of the chosen option(s)
- **Domain**: Abstract and diagrammatic logical reasoning

## Key Features

- 448 human-annotated visual multiple-choice questions collected from aptitude and reasoning tests
- Five reasoning skills used as subsets: inductive, deductive, numerical, spatial and mechanical
- Answer options live in the image and their label range varies per question (typically A-D or A-E, up to A-I)
- A handful of questions are multi-select (e.g. "which two proposals complete the diagram"), whose ground truth is a set of labels

## Evaluation Notes

- Default evaluation uses the **test** split and reports **Accuracy** overall and per reasoning skill
- Chain-of-thought prompting is used; the label is read from the final `ANSWER:` line and multi-select answers are compared as an unordered set, matching the official scoring rule
- Allow a generous `max_tokens`: when a reply is truncated before its `ANSWER:` line, the label is recovered from the last capital letter of the reply, which is a lenient guess
- Two of the released 448 items cannot be scored as published: `v1_382` carries neither a question nor an answer and is skipped, and `v1_20` labels its options with digits, which the letter-based answer parser cannot match — the reference implementations behave the same way


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `logic_vista` |
| **Dataset ID** | [evalscope/LogicVista](https://modelscope.cn/datasets/evalscope/LogicVista/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2407.04973) |
| **Tags** | `MCQ`, `MultiModal`, `Reasoning` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 447 |
| Prompt Length (Mean) | 529.27 chars |
| Prompt Length (Min/Max) | 397 / 900 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `inductive` | 107 | 463.01 | 406 | 645 |
| `deductive` | 93 | 577.96 | 426 | 790 |
| `numerical` | 95 | 582.38 | 460 | 811 |
| `spatial` | 78 | 450.85 | 397 | 747 |
| `mechanical` | 74 | 578.35 | 450 | 900 |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 447 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 156x165 - 1328x1352 |
| Formats | png |


## Sample Example

**Subset**: `inductive`

```json
{
  "input": [
    {
      "id": "3a2564ac",
      "content": [
        {
          "image": "[BASE64_IMAGE: png, ~46.7KB]"
        },
        {
          "text": "Answer the following multiple choice question. The answer options are shown in the image.\nThe last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the label of the option you choose. If more than one option is correct, list all of their labels on that line. Think step by step before answering.\n\nWhat choice (A, B, C, or D) should be in place of the question mark that fits the pattern?"
        }
      ]
    }
  ],
  "choices": [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I"
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "subset_key": "inductive",
  "metadata": {
    "id": "v1_0"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
Answer the following multiple choice question. The answer options are shown in the image.
The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the label of the option you choose. If more than one option is correct, list all of their labels on that line. Think step by step before answering.

{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets logic_vista \
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
    datasets=['logic_vista'],
    dataset_args={
        'logic_vista': {
            # subset_list: ['inductive', 'deductive', 'numerical']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
