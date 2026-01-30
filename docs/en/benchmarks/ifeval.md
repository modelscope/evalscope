# IFEval


## Overview

IFEval (Instruction-Following Eval) is a benchmark for evaluating how well language models follow explicit, verifiable instructions. It contains prompts with specific formatting, content, or structural requirements that can be objectively verified.

## Task Description

- **Task Type**: Instruction Following Evaluation
- **Input**: Prompts with explicit, verifiable constraints
- **Output**: Response that follows all specified instructions
- **Constraint Types**: Format, length, keywords, structure, etc.

## Key Features

- ~500 prompts with 25 types of verifiable instructions
- Instructions are objectively checkable (not subjective)
- Examples: "write exactly 3 paragraphs", "include the word X", "use bullet points"
- Tests instruction comprehension and compliance
- No ambiguity in evaluation criteria

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Four metrics available:
  - `prompt_level_strict`: All instructions in prompt must be followed
  - `prompt_level_loose`: Some tolerance for minor deviations
  - `inst_level_strict`: Per-instruction accuracy (strict)
  - `inst_level_loose`: Per-instruction accuracy (loose)
- `prompt_level_strict` is the primary metric
- Automatic verification of instruction compliance


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ifeval` |
| **Dataset ID** | [opencompass/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary) |
| **Paper** | N/A |
| **Tags** | `InstructionFollowing` |
| **Metrics** | `prompt_level_strict`, `inst_level_strict`, `prompt_level_loose`, `inst_level_loose` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 541 |
| Prompt Length (Mean) | 210.75 chars |
| Prompt Length (Min/Max) | 53 / 1858 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "cb71907f",
      "content": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "key": 1000,
    "prompt": "Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.",
    "instruction_id_list": [
      "punctuation:no_comma",
      "detectable_format:number_highlighted_sections",
      "length_constraints:number_words"
    ],
    "kwargs": [
      {
        "num_highlights": null,
        "relation": null,
        "num_words": null,
        "num_placeholders": null,
        "prompt_to_repeat": null,
        "num_bullets": null,
        "section_spliter": null,
        "num_sections": null,
        "capital_relation": null,
        "capital_frequency": null,
        "keywords": null,
        "num_paragraphs": null,
        "language": null,
        "let_relation": null,
        "letter": null,
        "let_frequency": null,
        "end_phrase": null,
        "forbidden_words": null,
        "keyword": null,
        "frequency": null,
        "num_sentences": null,
        "postscript_marker": null,
        "first_word": null,
        "nth_paragraph": null
      },
      {
        "num_highlights": 3,
        "relation": null,
        "num_words": null,
        "num_placeholders": null,
        "prompt_to_repeat": null,
        "num_bullets": null,
        "section_spliter": null,
        "num_sections": null,
        "capital_relation": null,
        "capital_frequency": null,
        "keywords": null,
        "num_paragraphs": null,
        "language": null,
        "let_relation": null,
        "letter": null,
        "let_frequency": null,
        "end_phrase": null,
        "forbidden_words": null,
        "keyword": null,
        "frequency": null,
        "num_sentences": null,
        "postscript_marker": null,
        "first_word": null,
        "nth_paragraph": null
      },
      {
        "num_highlights": null,
        "relation": "at least",
        "num_words": 300,
        "num_placeholders": null,
        "prompt_to_repeat": null,
        "num_bullets": null,
        "section_spliter": null,
        "num_sections": null,
        "capital_relation": null,
        "capital_frequency": null,
        "keywords": null,
        "num_paragraphs": null,
        "language": null,
        "let_relation": null,
        "letter": null,
        "let_frequency": null,
        "end_phrase": null,
        "forbidden_words": null,
        "keyword": null,
        "frequency": null,
        "num_sentences": null,
        "postscript_marker": null,
        "first_word": null,
        "nth_paragraph": null
      }
    ]
  }
}
```

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ifeval \
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
    datasets=['ifeval'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


