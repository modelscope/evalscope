# JobBench


## Overview

JobBench evaluates agentic systems on realistic professional work tasks that require reading reference files, producing
deliverables, and reconciling multi-source information. This adapter uses the ModelScope dataset
`evalscope/job-bench`.

## Task Description

- **Task Type**: Agentic professional work / deliverable generation
- **Input**: Workplace-style task prompt with optional reference files under `reference_files/`
- **Output**: Final deliverable files written to `jobbench_output/`
- **Dataset**: ModelScope `evalscope/job-bench`
- **Metric**: Weighted LLM-judge rubric score (`normalized_score`)

## Evaluation Notes

- The default evaluation split is `main`.
- Configure `judge_model_args` for rubric scoring.
- `normalized_score` is the primary weighted score (`total_score / max_score`); `pass_rate` is the unweighted
  proportion of fully passed rubrics; `total_score` is the raw sum of passed rubric weights.
- Docker runs use `python:3.11-slim-bookworm` by default. For formal evaluation, provide an image with the Office,
  PDF, and spreadsheet tools required by the tasks.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `job_bench` |
| **Dataset ID** | [evalscope/job-bench](https://modelscope.cn/datasets/evalscope/job-bench/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `Knowledge`, `MultiTurn` |
| **Metrics** | `normalized_score`, `pass_rate`, `total_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `main` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 65 |
| Prompt Length (Mean) | 3344.32 chars |
| Prompt Length (Min/Max) | 2000 / 4920 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "203fd1aa",
      "content": "You are preparing the Statistical Analysis Plan for Phase III trial XR-2847 evaluating cardiovascular disease prevention. The sponsor requires validation of statistical assumptions against empirical data and regulatory alignment before protoc ... [TRUNCATED 2026 chars] ... erables.\n\nWrite every final deliverable file under `jobbench_output`. Do not put intermediate scratch files there.\nYour final message may summarize what you produced, but files requested by the task must be actual files in\n`jobbench_output`.\n"
    }
  ],
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "bash",
      "description": "Execute a bash command inside the sandbox environment. Returns the combined stdout / stderr output of the command.",
      "parameters": {
        "properties": {
          "command": {
            "type": "string",
            "description": "The bash command to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "command"
        ]
      }
    },
    {
      "name": "python_exec",
      "description": "Execute Python source code inside the sandbox environment. Returns stdout and stderr output.",
      "parameters": {
        "properties": {
          "code": {
            "type": "string",
            "description": "Python source code to execute."
          },
          "timeout": {
            "type": "number",
            "description": "Maximum execution time in seconds (default: 60).",
            "default": 60
          }
        },
        "required": [
          "code"
        ]
      }
    }
  ],
  "metadata": {
    "task_id": "biostatisticians__task1",
    "reference_files": [
      "dataset/biostatisticians/task1/task_folder/Clinical_Study_Proposal_CVD_Prevention.csv",
      "dataset/biostatisticians/task1/task_folder/framingham.csv"
    ],
    "rubric_json": "{\n  \"rubrics\": [\n    {\n      \"rubric\": \"Does the analysis calculate the observed 10-year CHD event rate in the eligible population as 20.2% (236/1166) and identify this as higher than the assumed 15% control group rate?\",\n      \"weight\": 10,\n ... [TRUNCATED 4642 chars] ... ge criterion and by the sysBP criterion separately\",\n        \"The flow shows the final eligible population count (1,166)\",\n        \"The flow enables verification of the filtering process and assessment of generalizability\"\n      ]\n    }\n  ]\n}"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets job_bench \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":250}' \
    --limit 10  # Remove this line for formal evaluation
```

### Using Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['job_bench'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=250,
    ),
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
