# IFBench


## Overview

IFBench is a benchmark designed to evaluate how reliably AI models follow novel, challenging, and diverse verifiable instructions, with a strong focus on out-of-domain generalization. Developed by AllenAI, it addresses overfitting and data contamination issues in existing benchmarks.

## Task Description

- **Task Type**: Instruction Following Evaluation
- **Input**: Prompts with verifiable constraints
- **Output**: Responses that must satisfy specific constraints
- **Focus**: Precise instruction-following capabilities

## Key Features

- 58 manually curated verifiable constraints
- Categories: counting, formatting, word usage, etc.
- Focus on out-of-domain generalization
- Programmatic verification of constraint satisfaction
- Addresses data contamination concerns

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose
- Requires emoji, syllapy, and spacy packages
- Evaluates both strict and loose constraint satisfaction


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `ifbench` |
| **Dataset ID** | [allenai/IFBench_test](https://modelscope.cn/datasets/allenai/IFBench_test/summary) |
| **Paper** | N/A |
| **Tags** | `InstructionFollowing` |
| **Metrics** | `prompt_level_strict`, `inst_level_strict`, `prompt_level_loose`, `inst_level_loose` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 300 |
| Prompt Length (Mean) | 343.41 chars |
| Prompt Length (Min/Max) | 50 / 904 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "9e0a5835",
      "content": "What should the world's smartest man, surrounded by corruption, greed, inequity, madness, inequality, an establishment who preached conspiracy theories and wild speculations over truth and an equally evil resistance funded by the mega rich, a ... [TRUNCATED] ... ad here. Include keyword kaleidoscope once in your response, keyword nebula twice in your response, keyword whisper three times in your response, keyword labyrinth five times in your response, and keyword paradox seven times in your response."
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "key": "0",
    "prompt": "What should the world's smartest man, surrounded by corruption, greed, inequity, madness, inequality, an establishment who preached conspiracy theories and wild speculations over truth and an equally evil resistance funded by the mega rich, a ... [TRUNCATED] ... ad here. Include keyword kaleidoscope once in your response, keyword nebula twice in your response, keyword whisper three times in your response, keyword labyrinth five times in your response, and keyword paradox seven times in your response.",
    "instruction_id_list": [
      "count:keywords_multiple"
    ],
    "kwargs": [
      {
        "N": null,
        "capital_frequency": null,
        "capital_relation": null,
        "end_phrase": null,
        "first_word": null,
        "forbidden_words": null,
        "frequency": null,
        "keyword": null,
        "keyword1": "kaleidoscope",
        "keyword2": "nebula",
        "keyword3": "whisper",
        "keyword4": "labyrinth",
        "keyword5": "paradox",
        "keywords": null,
        "language": null,
        "let_frequency": null,
        "let_relation": null,
        "letter": null,
        "m": null,
        "max_words": null,
        "min_words": null,
        "n": null,
        "n_end": null,
        "n_start": null,
        "nth_paragraph": null,
        "num_bullets": null,
        "num_highlights": null,
        "num_paragraphs": null,
        "num_placeholders": null,
        "num_sections": null,
        "num_sentences": null,
        "num_words": null,
        "options": null,
        "percentage": null,
        "postscript_marker": null,
        "prompt_to_repeat": null,
        "reference_text": null,
        "relation": null,
        "section_spliter": null,
        "sep": null,
        "small_n": null,
        "word": null
      }
    ]
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

*No prompt template defined.*

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets ifbench \
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
    datasets=['ifbench'],
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


