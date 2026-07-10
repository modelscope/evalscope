# ResearchRubrics


## Overview

ResearchRubrics evaluates Deep Research agents on realistic, open-ended research tasks. Each task pairs a user prompt
with expert-written, fine-grained rubrics covering explicit and implicit requirements, information synthesis,
references, communication quality, and instruction following.

## Task Description

- **Task Type**: Multi-turn research agent / long-form report generation
- **Input**: One open-ended research prompt
- **Output**: A Markdown research report produced after iterative tool use
- **Dataset**: 101 tasks and 2,593 weighted rubric criteria
- **Metric**: Binary rubric compliance score

## Agent Runtime

- Uses EvalScope's built-in AgentLoop with the ``function_calling`` strategy and a ``bash`` tool by default.
- The default bash tool runs in a per-sample temporary directory through ``LocalAgentEnvironment`` and uses the host
  network. This environment is not a security sandbox: absolute paths can still access the host filesystem. Do not use
  the default runtime with untrusted models on shared or sensitive machines.
- ``dataset_args.extra_params.strategy`` can be set to ``react``. Both built-in strategies require native function
  calling; ReAct additionally injects a Think -> Act -> Observation system prompt.
- Optional MCP servers from ``NativeAgentConfig.mcp_servers`` are merged with bash. ``ExternalAgentConfig`` routes the
  prompt through EvalScope's external agent bridge instead.
- For this benchmark-owned AgentLoop, strategy and max steps are configured through ``dataset_args.extra_params``;
  corresponding fields on ``NativeAgentConfig`` are not used.
- If the native loop exhausts ``max_steps`` while still calling tools, the benchmark makes one final tool-free model
  call so the gathered research is preserved as a reviewable Markdown report.

## Evaluation Notes

- ResearchRubrics requires ``judge_model_args`` and ``judge_strategy='auto'`` or ``'llm'``. Gemini 2.5 Pro is the
  recommended judge for comparison with the paper, but no provider or model is hard-coded.
- Every rubric is graded independently as Satisfied (1) or Not Satisfied (0), matching the public binary grader. The
  paper's ternary scores are not directly comparable.
- Negative-weight criteria subtract from the numerator when the undesirable behavior is present. Scores are not
  clipped.
- Long reports are evaluated with the official chunk-evidence-synthesis approach when they exceed the configured judge
  context threshold.
- A full run performs 2,593 rubric evaluations and can be expensive. Current-events tasks are also sensitive to the
  date and web sources available at evaluation time.

## Configuration

- ``strategy``: ``function_calling`` (default) or ``react``
- ``max_steps``: 50 by default
- ``judge_context_limit``: 150,000 estimated tokens
- ``judge_chunk_size``: 100,000 estimated tokens
- ``judge_retries``: 3 attempts per judge request

The judge must be configured explicitly. For example:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['researchrubrics'],
    judge_strategy='llm',
    judge_model_args={
        'model_id': 'YOUR_JUDGE_MODEL',
        'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
        'api_key': 'YOUR_JUDGE_API_KEY',
        'generation_config': {'temperature': 0.0},
    },
    limit=1,
))
```

Resources: [Paper](https://arxiv.org/abs/2511.07685) |
[GitHub](https://github.com/scaleapi/researchrubrics) |
[Dataset](https://modelscope.cn/datasets/evalscope/researchrubrics)


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `researchrubrics` |
| **Dataset ID** | [evalscope/researchrubrics](https://modelscope.cn/datasets/evalscope/researchrubrics/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2511.07685) |
| **Tags** | `Agent`, `MultiTurn`, `Reasoning`, `Retrieval` |
| **Metrics** | `compliance_score` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 101 |
| Prompt Length (Mean) | 555.35 chars |
| Prompt Length (Min/Max) | 102 / 1747 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "aeaec74a",
      "content": "I want to create a plan for July 4, 2025, i.e., Independence Day in Washington DC. I would like an itinerary of all the things to do and all the activities that are planned for Independence Day. Create a plan for the whole day and also extend it to the weekend, if required. Provide some reviews or explain why one should visit the place or engage in the activity. Add any additional information that is required."
    }
  ],
  "target": "[{\"criterion\": \"The response covers the period from 9:00 AM or earlier through at least 10:00 PM on 4 July 2025\", \"weight\": 5.0, \"axis\": \"Explicit Criteria\"}, {\"criterion\": \"The response contains clear section headers for parts of the schedul ... [TRUNCATED 4470 chars] ...  events from years other than 2025 (e.g., seeing a miltiary parade, information about \\\"A Capital Fourth\\\" for 2024, information the parade route for 2023, referencing a cancellation from 2020).\", \"weight\": -4.0, \"axis\": \"Explicit Criteria\"}]",
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
    }
  ],
  "metadata": {
    "sample_id": "6847465956a0f6376a605427",
    "domain": "Current Events",
    "conceptual_breadth": "Simple",
    "logical_nesting": "Intermediate",
    "exploration": "Medium"
  }
}
```

*Note: Some content was truncated for display.*

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `function_calling` | Agent strategy used by the built-in AgentLoop. Choices: ['function_calling', 'react'] |
| `max_steps` | `int` | `50` | Maximum number of agent steps per sample. |
| `judge_context_limit` | `int` | `150000` | Estimated token limit before rubric judging switches to chunking. |
| `judge_chunk_size` | `int` | `100000` | Maximum estimated tokens in each document chunk sent to the judge. |
| `judge_retries` | `int` | `3` | Maximum attempts for each rubric judge request and JSON parse. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets researchrubrics \
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
    datasets=['researchrubrics'],
    dataset_args={
        'researchrubrics': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
