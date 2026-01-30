# DrivelologyNarrativeSelection

## Overview

Drivelology Narrative Selection evaluates models' ability to understand the underlying narrative of "drivelology" text - linguistic utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.

## Task Description

- **Task Type**: Multiple-Choice Narrative Understanding
- **Input**: Drivelology text with multiple narrative interpretation options
- **Output**: Best option representing the underlying narrative
- **Domain**: Linguistic analysis, narrative comprehension

## Key Features

- Tests deep narrative understanding
- Requires interpretation of layered meanings
- Multiple-choice format with challenging distractors
- Easy and hard difficulty levels
- Tests cultural and contextual understanding

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Simple accuracy metric
- Subsets: multiple-choice-english-easy, multiple-choice-english-hard

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `drivel_selection` |
| **Dataset ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **Paper** | N/A |
| **Tags** | `MCQ` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,200 |
| Prompt Length (Mean) | 1413.26 chars |
| Prompt Length (Min/Max) | 754 / 2865 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `multiple-choice-english-easy` | 600 | 1563.53 | 908 | 2865 |
| `multiple-choice-english-hard` | 600 | 1262.98 | 754 | 2348 |

## Sample Example

**Subset**: `multiple-choice-english-easy`

```json
{
  "input": [
    {
      "id": "44908073",
      "content": "Tell me the best option in the following options which represents the underlying narrative of the text?\nThe entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C, ... [TRUNCATED] ...  be achieved by simply resting today and tomorrow. It humorously implies that diligence is overrated and taking breaks is the key to progress. This narrative undermines the importance of effort, presenting relaxation as the ultimate solution."
    }
  ],
  "choices": [
    "The passage reflects on the inevitability of fate, stating that what happens the day after tomorrow is beyond our control. Therefore, it encourages living in the moment and enjoying today and tomorrow. It conveys a message of mindfulness and acceptance.",
    "This creates a paradoxical tone, as it acknowledges the value of diligence but simultaneously advocates for procrastination. The underlying message could reflect a lighthearted take on balancing work and rest or even poking fun at the tendency to delay responsibilities.",
    "The text discusses the cyclical nature of time, arguing that the day after tomorrow holds the key to breaking free from monotony. It proposes resting today and tomorrow to prepare for this transformative moment. This symbolizes renewal and the anticipation of change.",
    "The text emphasizes the importance of teamwork, suggesting that collective effort tomorrow will yield the best results. It then humorously advises everyone to take a break today to gather energy. This highlights the value of preparation over immediate action.",
    "The text suggests that hard work is unnecessary, as success can be achieved by simply resting today and tomorrow. It humorously implies that diligence is overrated and taking breaks is the key to progress. This narrative undermines the importance of effort, presenting relaxation as the ultimate solution."
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "metadata": {}
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Tell me the best option in the following options which represents the underlying narrative of the text?
The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}.

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
    --datasets drivel_selection \
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
    datasets=['drivel_selection'],
    dataset_args={
        'drivel_selection': {
            # subset_list: ['multiple-choice-english-easy', 'multiple-choice-english-hard']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


