# Multi-IF


## Overview

Multi-IF is a benchmark designed to evaluate LLM capabilities in multi-turn instruction following within a multilingual environment. It tests the ability to follow complex instructions across multiple conversation turns in different languages.

## Task Description

- **Task Type**: Multi-Turn Multilingual Instruction Following
- **Input**: Multi-turn conversation with instructions
- **Output**: Responses following given instructions
- **Domains**: Instruction following, multilingual understanding

## Key Features

- 11 supported languages: Chinese, English, German, Italian, Vietnamese, Spanish, Hindi, Portuguese, French, Thai, Russian
- Multi-turn conversation evaluation (up to 3 turns)
- Tests instruction following in multilingual contexts
- Both strict and loose evaluation metrics
- Prompt-level and instruction-level scoring

## Evaluation Notes

- Default evaluation uses the **train** split
- Configurable max_turns (1-3, default: 3)
- Four metrics tracked:
  - `prompt_level_strict/loose`: Strict/loose prompt-level accuracy
  - `inst_level_strict/loose`: Strict/loose instruction-level accuracy
- Requires: nltk, langdetect, emoji (for Chinese), pythainlp (for Thai)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `multi_if` |
| **Dataset ID** | [facebook/Multi-IF](https://modelscope.cn/datasets/facebook/Multi-IF/summary) |
| **Paper** | N/A |
| **Tags** | `InstructionFollowing`, `MultiLingual`, `MultiTurn` |
| **Metrics** | `prompt_level_strict`, `inst_level_strict`, `prompt_level_loose`, `inst_level_loose` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4,501 |
| Prompt Length (Mean) | 0 chars |
| Prompt Length (Min/Max) | 0 / 0 chars |

**Per-Subset Statistics:**

| Subset | Samples | Prompt Mean | Prompt Min | Prompt Max |
|--------|---------|-------------|------------|------------|
| `Chinese` | 454 | 0 | 0 | 0 |
| `English` | 909 | 0 | 0 | 0 |
| `Italian` | 493 | 0 | 0 | 0 |
| `Spanish` | 516 | 0 | 0 | 0 |
| `Hindi` | 542 | 0 | 0 | 0 |
| `Portuguese` | 524 | 0 | 0 | 0 |
| `French` | 548 | 0 | 0 | 0 |
| `Russian` | 515 | 0 | 0 | 0 |

## Sample Example

**Subset**: `Chinese`

```json
{
  "input": [
    {
      "id": "cb0c68fe",
      "content": ""
    }
  ],
  "target": "",
  "id": 0,
  "group_id": 0,
  "subset_key": "Chinese",
  "metadata": {
    "turns": null,
    "responses": null,
    "turn_1_prompt": "{\"role\": \"user\", \"content\": \"\\u5199\\u4e00\\u4e2a300+\\u5b57\\u7684\\u603b\\u7ed3\\u5173\\u4e8e\\u7ef4\\u57fa\\u767e\\u79d1\\u9875\\u9762\\\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\\\"\\uff0c\\u4e0d\\u8981\\u4f7f\\u7528\\u4efb\\u4f55\\u9017\\u53f7\\uff0c\\u5e76\\u4e14\\u81f3\\u5c11\\u7528markdown\\u683c\\u5f0f\\u7a81\\u51fa3\\u4e2a\\u6709\\u6807\\u9898\\u7684\\u90e8\\u5206\\uff0c\\u4f8b\\u5982*\\u7a81\\u51fa\\u90e8\\u52061*\\uff0c*\\u7a81\\u51fa\\u90e8\\u52062*\\uff0c*\\u7a81\\u51fa\\u90e8\\u52063*\\u3002\"}",
    "turn_1_instruction_id_list": "[\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\"]",
    "turn_1_kwargs": "[\"{}\", \"{\\\"num_highlights\\\": 3}\", \"{\\\"relation\\\": \\\"at least\\\", \\\"num_words\\\": 300}\"]",
    "turn_2_prompt": "{\"role\": \"user\", \"content\": \"\\u4f60\\u7684\\u56de\\u7b54\\u5e94\\u8be5\\u5305\\u542b\\u4ee5\\u4e0b\\u5173\\u952e\\u8bcd\\uff1a\\u5341\\u5b57\\u519b\\uff0c\\u9ece\\u5df4\\u5ae9\\uff0c\\u7a46\\u65af\\u6797\\u3002\"}",
    "turn_2_instruction_id_list": "[\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\", \"keywords:existence\"]",
    "turn_2_kwargs": "[\"{}\", \"{\\\"num_highlights\\\": 3}\", \"{\\\"relation\\\": \\\"at least\\\", \\\"num_words\\\": 300}\", \"{\\\"keywords\\\": [\\\"\\\\u5341\\\\u5b57\\\\u519b\\\", \\\"\\\\u9ece\\\\u5df4\\\\u5ae9\\\", \\\"\\\\u7a46\\\\u65af\\\\u6797\\\"]}\"]",
    "turn_3_prompt": "{\"role\": \"user\", \"content\": \"\\u4f60\\u7684\\u56de\\u7b54\\u5e94\\u8be5\\u4ee5\\u201c\\u8fd9\\u4e2a\\u6982\\u8ff0\\u63d0\\u4f9b\\u4e86\\u5341\\u5b57\\u519b\\u65f6\\u4ee3\\u653f\\u6cbb\\u548c\\u6218\\u4e89\\u7684\\u590d\\u6742\\u6027\\u548c\\u9634\\u8c0b\\u7684\\u6982\\u89c8\\u3002\\u201d\\u8fd9\\u4e2a\\u786e\\u5207\\u7684\\u77ed\\u8bed\\u7ed3\\u5c3e\\uff0c\\u4e0d\\u5141\\u8bb8\\u6709\\u5176\\u4ed6\\u6587\\u5b57\\u51fa\\u73b0\\u5728\\u8fd9\\u4e2a\\u77ed\\u8bed\\u540e\\u9762\\u3002\"}",
    "turn_3_instruction_id_list": "[\"punctuation:no_comma\", \"detectable_format:number_highlighted_sections\", \"length_constraints:number_words\", \"keywords:existence\", \"startend:end_checker\"]",
    "turn_3_kwargs": "[\"{}\", \"{\\\"num_highlights\\\": 3}\", \"{\\\"relation\\\": \\\"at least\\\", \\\"num_words\\\": 300}\", \"{\\\"keywords\\\": [\\\"\\\\u5341\\\\u5b57\\\\u519b\\\", \\\"\\\\u9ece\\\\u5df4\\\\u5ae9\\\", \\\"\\\\u7a46\\\\u65af\\\\u6797\\\"]}\", \"{\\\"end_phrase\\\": \\\"\\\\u8fd9\\\\u4e2a\\\\u6982\\\\u8ff0\\\\u63d0\\\\u4f9b\\\\u4e86\\\\u5341\\\\u5b57\\\\u519b\\\\u65f6\\\\u4ee3\\\\u653f\\\\u6cbb\\\\u548c\\\\u6218\\\\u4e89\\\\u7684\\\\u590d\\\\u6742\\\\u6027\\\\u548c\\\\u9634\\\\u8c0b\\\\u7684\\\\u6982\\\\u89c8\\\\u3002\\\"}\"]",
    "key": "1000:1:zh",
    "turn_index": 0,
    "language": "Chinese"
  }
}
```

## Prompt Template

*No prompt template defined.*

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_turns` | `int` | `3` | Maximum number of interactive turns to evaluate (1-3). Choices: [1, 2, 3] |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets multi_if \
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
    datasets=['multi_if'],
    dataset_args={
        'multi_if': {
            # subset_list: ['Chinese', 'English', 'Italian']  # optional, evaluate specific subsets
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


