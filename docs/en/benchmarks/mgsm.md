# MGSM


## Overview

MGSM (Multilingual Grade School Math) is a benchmark designed to evaluate multilingual mathematical reasoning capabilities of language models. It extends GSM8K to 11 typologically diverse languages, testing whether models can perform chain-of-thought reasoning across different languages.

## Task Description

- **Task Type**: Multilingual Mathematical Word Problem Solving
- **Input**: Grade school math word problem in one of 11 languages
- **Output**: Step-by-step reasoning with numerical answer
- **Languages**: English, Spanish, French, German, Russian, Chinese, Japanese, Thai, Swahili, Bengali, Telugu

## Key Features

- 250 problems per language (translated from GSM8K)
- 11 typologically diverse languages covering different language families
- Tests multilingual chain-of-thought reasoning capabilities
- Same problem content across languages for cross-lingual comparison
- Designed to evaluate language-agnostic mathematical reasoning

## Evaluation Notes

- Default configuration uses **4-shot** examples
- Answers should be formatted within `\boxed{}` for proper extraction
- Use `subset_list` to evaluate specific languages (e.g., `['en', 'zh', 'ja']`)
- Cross-lingual performance comparison supported
- Few-shot examples are drawn from the train split in the same language


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mgsm` |
| **Dataset ID** | [evalscope/mgsm](https://modelscope.cn/datasets/evalscope/mgsm/summary) |
| **Paper** | N/A |
| **Tags** | `Math`, `MultiLingual`, `Reasoning` |
| **Metrics** | `acc` |
| **Default Shots** | 4-shot |
| **Evaluation Split** | `test` |
| **Train Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 2,750 |
| Prompt Length (Mean) | 1742.98 chars |
| Prompt Length (Min/Max) | 791 / 2464 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `en` | 250 | 1790.71 | 1637 | 2165 |
| `es` | 250 | 1940.02 | 1773 | 2371 |
| `fr` | 250 | 2047 | 1878 | 2440 |
| `de` | 250 | 1963.9 | 1792 | 2386 |
| `ru` | 250 | 1831.66 | 1667 | 2214 |
| `zh` | 250 | 842.16 | 791 | 946 |
| `ja` | 250 | 1102.33 | 1035 | 1248 |
| `th` | 250 | 1835.53 | 1699 | 2135 |
| `sw` | 250 | 1953.48 | 1780 | 2354 |
| `bn` | 250 | 1759.28 | 1601 | 2106 |
| `te` | 250 | 2106.77 | 1939 | 2464 |

## Sample Example

**Subset**: `en`

```json
{
  "input": [
    {
      "id": "d67cb3cf",
      "content": "Here are some examples of how to solve similar problems:\n\nQuestion: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?\n\nReasoning:\nStep-by-Step Answer: Roger sta ... [TRUNCATED] ...  every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}.\n\n"
    }
  ],
  "target": "18",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "reasoning": null,
    "equation_solution": null
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
Please reason step by step, and put your final answer within \boxed{{}}.


```

<details>
<summary>Few-shot Template</summary>

```text
Here are some examples of how to solve similar problems:

{fewshot}

{question}
Please reason step by step, and put your final answer within \boxed{{}}.


```

</details>

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mgsm \
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
    datasets=['mgsm'],
    dataset_args={
        'mgsm': {
            # subset_list: ['en', 'es', 'fr']  # optional, evaluate specific subsets
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


