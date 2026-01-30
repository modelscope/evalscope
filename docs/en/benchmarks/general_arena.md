# GeneralArena


## Overview

GeneralArena is a custom benchmark designed to evaluate the performance of large language models in a competitive setting, where models are pitted against each other in custom tasks to determine their relative strengths and weaknesses.

## Task Description

- **Task Type**: Model vs Model Arena Evaluation
- **Input**: User prompts with responses from multiple models
- **Output**: Pairwise comparison judgments and ELO ratings
- **Focus**: Comparative model evaluation

## Key Features

- LLM-as-judge for pairwise comparisons
- ELO rating calculation for model ranking
- Supports custom model pairs for comparison
- Winrate and ranking metrics
- Configurable baseline model

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Requires model outputs from previous evaluations
- Uses LLM-as-judge for quality assessment
- Aggregation: ELO rating calculation
- See [Arena User Guide](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html) for details


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `general_arena` |
| **Dataset ID** | `general_arena` |
| **Paper** | N/A |
| **Tags** | `Arena`, `Custom` |
| **Metrics** | `winrate` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `test` |
| **Aggregation** | `elo` |


## Data Statistics

*Statistics not available.*

## Sample Example

*Sample example not available.*

## Prompt Template

**System Prompt:**
```text
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".
```

**Prompt Template:**
```text
<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_1}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_2}
<|The End of Assistant B's Answer|>
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[dict]` | `[{'name': 'qwen-plus', 'report_path': 'outputs/20250627_172550/reports/qwen-plus'}, {'name': 'qwen2.5-7b', 'report_path': 'outputs/20250627_172817/reports/qwen2.5-7b-instruct'}]` | List of model entries with name and report_path for arena comparison. |
| `baseline` | `str` | `qwen2.5-7b` | Baseline model name used for ELO and winrate comparisons. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets general_arena \
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
    datasets=['general_arena'],
    dataset_args={
        'general_arena': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


