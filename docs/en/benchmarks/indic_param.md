# IndicParam


## Overview

IndicParam is a graduate-level benchmark evaluating LLM understanding of low- and extremely
low-resource Indic languages. All 13,207 multiple-choice questions are sourced from official UGC-NET
language question papers and answer keys, presented in each language's native script (or code-mixed
form for Sanskrit-English).

## Task Description

- **Task Type**: Graduate-Level Multiple-Choice Question Answering
- **Input**: A UGC-NET exam question with 4 answer choices, in a low-resource Indic language
- **Output**: Correct answer letter
- **Languages**: Bodo, Dogri, Gujarati (Surya script), Konkani, Maithili, Marathi, Nepali, Oriya,
  Rajasthani, Sanskrit, Sanskrit-English code-mixed, Santali

## Key Features

- 13,207 multiple-choice questions sourced from official UGC-NET language question papers
- 12 low-resource Indic languages/scripts, including extremely low-resource ones like Bodo and Santali
- Questions are presented in each language's native script (or code-mixed form for Sanskrit-English)
- All languages ship in a single dataset config, differentiated by the `subject` field

## Evaluation Notes

- Default configuration uses **0-shot** evaluation (test split, the only split available)
- Use `subset_list` to evaluate specific languages
- All languages ship in a single dataset config, differentiated by the `subject` field; this adapter
  reformats by that field


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `indic_param` |
| **Dataset ID** | [bharatgenai/IndicParam](https://modelscope.cn/datasets/bharatgenai/IndicParam/summary) |
| **Paper** | N/A |
| **Tags** | `Knowledge`, `MCQ`, `MultiLingual` |
| **Metrics** | `accuracy` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 13,207 |
| Prompt Length (Mean) | 376.02 chars |
| Prompt Length (Min/Max) | 218 / 1413 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Bodo` | 1,313 | 461.37 | 256 | 738 |
| `Dogri` | 1,027 | 487.72 | 245 | 853 |
| `Gujarati_surya` | 1,044 | 395.79 | 255 | 611 |
| `Konkani` | 1,328 | 396.77 | 245 | 1413 |
| `Maithili` | 1,286 | 284.67 | 218 | 451 |
| `Marathi` | 1,245 | 382.66 | 242 | 957 |
| `Nepali` | 1,038 | 406.12 | 260 | 857 |
| `Oriya` | 577 | 365.04 | 239 | 924 |
| `Rajasthani` | 1,190 | 321.32 | 237 | 1136 |
| `Sanskrit` | 1,315 | 304.51 | 229 | 833 |
| `Sanskrit Mix` | 971 | 352.41 | 253 | 693 |
| `Santali` | 873 | 366.16 | 233 | 809 |

## Sample Example

**Subset**: `Bodo`

```json
{
  "input": [
    {
      "id": "0616580a",
      "content": "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D.\n\nआथिखालाव सुबुं थुनलाइफोरखौ बुथुमनो थाखाय बबे आदबखौ रासिनै बाहायनाय जायो\n\nA) फट' दैखांनाय\nB) रेकरडिं खालामनाय\nC) सल बुंहोनाय\nD) सल खोनासंनाय"
    }
  ],
  "choices": [
    "फट' दैखांनाय",
    "रेकरडिं खालामनाय",
    "सल बुंहोनाय",
    "सल खोनासंनाय"
  ],
  "target": "B",
  "id": 0,
  "group_id": 0,
  "subset_key": "Bodo",
  "metadata": {
    "subject": "Bodo",
    "exam_name": "Question Papers of NET Dec. 2012 Bodo Paper III hindi"
  }
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
    --datasets indic_param \
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
    datasets=['indic_param'],
    dataset_args={
        'indic_param': {
            # subset_list: ['Bodo', 'Dogri', 'Gujarati_surya']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
