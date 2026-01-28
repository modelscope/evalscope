# DocMath


## Overview

DocMath-Eval is a comprehensive benchmark focused on numerical reasoning within specialized domains. It requires models to comprehend long and specialized documents and perform numerical reasoning to answer questions.

## Task Description

- **Task Type**: Document-based Mathematical Reasoning
- **Input**: Long document context + numerical reasoning question
- **Output**: Numerical answer with reasoning
- **Focus**: Long-context comprehension and quantitative reasoning

## Key Features

- Long specialized documents requiring comprehension
- Numerical reasoning within document context
- Multiple complexity levels (comp/simp, long/short)
- Tests real-world document understanding
- Requires both reading comprehension and math skills

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Uses LLM-as-judge for answer evaluation
- Subsets: complong_testmini, compshort_testmini, simplong_testmini, simpshort_testmini
- Answer format: "Therefore, the answer is (answer)"


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `docmath` |
| **Dataset ID** | [yale-nlp/DocMath-Eval](https://modelscope.cn/datasets/yale-nlp/DocMath-Eval/summary) |
| **Paper** | N/A |
| **Tags** | `LongContext`, `Math`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 800 |
| Prompt Length (Mean) | 68791.03 chars |
| Prompt Length (Min/Max) | 505 / 1009038 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `complong_testmini` | 300 | 175355.17 | 18687 | 1009038 |
| `compshort_testmini` | 200 | 1990.74 | 505 | 9460 |
| `simplong_testmini` | 100 | 13972.84 | 6870 | 24001 |
| `simpshort_testmini` | 200 | 3154.2 | 560 | 9600 |

## Sample Example

**Subset**: `complong_testmini`

```json
{
  "input": [
    {
      "id": "a07cbfcf",
      "content": "Please read the following text and answer the question below.\n\n<text>\nDELTA AIR LINES, INC.\nConsolidated Balance Sheets\n| (in millions, except share data) | March 31, 2018 | December 31, 2017 |\n| ASSETS |\n| Current Assets: |\n| Cash and cash e ... [TRUNCATED] ... comprehensive income for foreign currency exchange contracts in 2017 and 2018, and the changes in value for derivative contracts and other in 2018, in million?\n\nFormat your response as follows: \"Therefore, the answer is (insert answer here)\"."
    }
  ],
  "target": "-31.0",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "question_id": "complong-testmini-0",
    "answer_type": "float"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
Please read the following text and answer the question below.

<text>
{context}
</text>

{question}

Format your response as follows: "Therefore, the answer is (insert answer here)".
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets docmath \
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
    datasets=['docmath'],
    dataset_args={
        'docmath': {
            # subset_list: ['complong_testmini', 'compshort_testmini', 'simplong_testmini']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


