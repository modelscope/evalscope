# CountQA


## Overview

CountQA probes object counting, a basic perceptual skill that multimodal models are largely
unevaluated on. Its images were hand-captured in everyday environments and deliberately feature
high object density, clutter and occlusion, so counting cannot be solved by detecting a handful of
well-separated objects.

## Task Description

- **Task Type**: Free-form Visual Question Answering (object counting)
- **Input**: A real-world photograph + a counting question (e.g. "How many jackets are there?")
- **Output**: A single integer
- **Domain**: Everyday scenes — groceries, kitchenware, tools, clothing, office and outdoor objects

## Key Features

- 1,528 question-answer pairs over 1,001 images; an image may carry several questions
- Ground-truth counts were annotated *in situ* during capture rather than post-hoc, and range from 0 to 400
- Questions include compositional ones that require summing over several object types
- Roughly half the images are cluttered rather than focused on a single subject (recorded as
  ``is_focused`` in each sample's metadata), and scene categories are recorded as ``categories``

## Evaluation Notes

- Default evaluation uses the **test** split as a single subset
- Primary metric: **Accuracy** (`accuracy`) — Exact Match against the ground-truth integer
- Secondary metric: **relaxed_acc** — the paper's Relaxed Accuracy, counting a prediction correct
  when it is within 5% of the ground truth
- The paper's system prompt is used as-is; it constrains the reply to a bare integer
- Answer parsing takes the reply if it is already an integer, otherwise its first integer — the
  rule the paper states for its rewriter LLM. A reply with no digit scores 0, so `max_tokens` must
  leave the model room to reach its answer; a model that narrates its count ("row 1 has 3 ...") is
  scored on the first number it mentions rather than on its stated total
- Scoring is deterministic arithmetic and needs no LLM judge: keep `judge.strategy` at `rule` or
  `auto`, since `llm` replaces both metrics with a generic judge score. To read a different number
  out of a model that ignores the output format, prepend a per-run filter such as
  `filters={'regex': {'regex_pattern': '(\d+)', 'group_select': -1}}` (last number) via
  `dataset_args` rather than editing the adapter
- [Paper](https://arxiv.org/abs/2508.06585)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `count_qa` |
| **Dataset ID** | [evalscope/CountQA](https://modelscope.cn/datasets/evalscope/CountQA/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2508.06585) |
| **Tags** | `MultiModal`, `QA`, `Reasoning` |
| **Metrics** | `accuracy`, `relaxed_acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**System Prompt:**
```text
You are a helpful assistant that counts the number of items in an image. The user will provide an image and ask a question about the number of a certain type of item in the image. If the user question is referring to multiple objects, it means that you need to provide a sum of the number of items. You will count the number of items and return the number as an integer. Your output should STRICTLY be a single integer and nothing else.
```

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets count_qa \
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
    datasets=['count_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
