# DrivelologyBinaryClassification

## Overview

Drivelology Binary Classification evaluates models' ability to identify "drivelology" - a unique linguistic phenomenon characterized as "nonsense with depth." These are utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive.

## Task Description

- **Task Type**: Binary Text Classification (Yes/No)
- **Input**: Text sample to classify
- **Output**: "Yes" if drivelology, "No" otherwise
- **Domain**: Linguistic analysis, humor detection, pragmatics

## Key Features

- Tests understanding of layered linguistic meanings
- Distinguishes nonsense-with-depth from pure nonsense and normal text
- Requires contextual understanding and emotional insight
- Covers humor, irony, sarcasm detection
- Multiple difficulty levels available

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score
- Subsets: binary-english-easy, binary-english-hard, binary-chinese-easy, binary-chinese-hard

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `drivel_binary` |
| **Dataset ID** | [extraordinarylab/drivel-hub](https://modelscope.cn/datasets/extraordinarylab/drivel-hub/summary) |
| **Paper** | N/A |
| **Tags** | `Yes/No` |
| **Metrics** | `accuracy`, `precision`, `recall`, `f1_score`, `yes_ratio` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `f1` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 1,200 |
| Prompt Length (Mean) | 1056.08 chars |
| Prompt Length (Min/Max) | 984 / 1449 chars |

## Sample Example

**Subset**: `binary-classification`

```json
{
  "input": [
    {
      "id": "ddbda8da",
      "content": [
        {
          "text": "#Instruction#:\nClassify whether the given text is a Drivelology sample or not.\n\n#Definition#:\n- Drivelology: Statements that appear logically coherent but contain deeper, often paradoxical meanings.\nThese challenge conventional interpretation ... [TRUNCATED] ... ology.\n\n#Output Format#:\nYou should try your best to answer \"Yes\" if the given input text is Drivelology, otherwise specify \"No\".\nThe answer you give MUST be \"Yes\" or \"No\"\".\n\n#Input Text#: A: Name? B: Henry. A: Age? B: E-N-R-Y.\n#Your Answer#:"
        }
      ]
    }
  ],
  "target": "YES",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "answer": "YES"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

<details>
<summary>Few-shot Template</summary>

```text
{question}
```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets drivel_binary \
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
    datasets=['drivel_binary'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


