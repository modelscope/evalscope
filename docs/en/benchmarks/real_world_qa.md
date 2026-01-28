# RealWorldQA


## Overview

RealWorldQA is a benchmark contributed by XAI designed to evaluate multimodal AI models' understanding of real-world spatial and physical environments. It uses authentic images from everyday scenarios to test practical visual comprehension.

## Task Description

- **Task Type**: Real-World Visual Question Answering
- **Input**: Real-world image with spatial/physical question
- **Output**: Verifiable answer about the scene
- **Domain**: Physical environments, driving scenarios, everyday scenes

## Key Features

- 700+ images from real-world scenarios
- Includes vehicle-captured images (driving scenes)
- Questions with verifiable ground-truth answers
- Tests spatial understanding and physical reasoning
- Evaluates practical AI understanding capabilities

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Answers should follow "ANSWER: [ANSWER]" format
- Uses step-by-step reasoning prompting
- Simple accuracy metric for evaluation
- Tests models on practical, real-world scenarios


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `real_world_qa` |
| **Dataset ID** | [lmms-lab/RealWorldQA](https://modelscope.cn/datasets/lmms-lab/RealWorldQA/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MultiModal`, `QA` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 765 |
| Prompt Length (Mean) | 554.79 chars |
| Prompt Length (Min/Max) | 459 / 904 chars |

**Image Statistics:**

| Metric | Value |
|--------|-------|
| Total Images | 765 |
| Images per Sample | min: 1, max: 1, mean: 1 |
| Resolution Range | 626x418 - 1536x1405 |
| Formats | webp |


## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "6492d8ea",
      "content": [
        {
          "text": "Read the picture and solve the following problem step by step.The last line of your response should be of the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem.\n\nIn which direction is the front wheel of the  ... [TRUNCATED] ... e letter of the correct option and nothing else.\n\nRemember to put your answer on its own line at the end in the form \"ANSWER: [ANSWER]\" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \\boxed command."
        },
        {
          "image": "[BASE64_IMAGE: webp, ~810.4KB]"
        }
      ]
    }
  ],
  "target": "C",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "image_path": "0.webp"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Read the picture and solve the following problem step by step.The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \boxed command.
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets real_world_qa \
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
    datasets=['real_world_qa'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


