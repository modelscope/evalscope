# DeepSearchQA

## Overview

DeepSearchQA is a Google DeepMind benchmark for evaluating deep research agents on difficult multi-step information-seeking tasks across the open web. It contains 900 prompts spanning 17 domains and is designed to measure exhaustive answer-set generation rather than single-answer retrieval alone.

## Task Description

- **Task Type**: Search-agent factual question answering
- **Input**: A natural-language research question
- **Output**: A single answer or complete answer set, depending on the question
- **Grading**: LLM-as-judge semantic matching against the gold answer and answer type

## Key Features

- Tests systematic collation of fragmented information from multiple sources
- Requires entity resolution and de-duplication for set-answer tasks
- Penalizes both under-retrieval and excessive/hallucinated answers
- Uses `problem_category` for analysis metadata; `answer_type` is withheld from the model during inference
- Compatible with EvalScope agent configurations for native or external web-capable agents

## Agent Tool Configuration

DeepSearchQA does not hard-code a search provider. By default it runs through EvalScope native AgentLoop without external search tools. To evaluate a web-capable agent, set `TaskConfig.agent_config` and attach the search/fetch tools that should be available to the model. If `NativeAgentConfig.max_steps` is omitted, DeepSearchQA uses its benchmark-level AgentLoop default of 30 steps.

See the [DeepSearchQA usage guide](https://evalscope.readthedocs.io/en/latest/third_party/deepsearchqa.html) for
runtime examples, MCP search/fetch configuration, and evaluation notes.

## Evaluation Notes

- EvalScope loads the ModelScope dataset `google/deepsearchqa` from the `eval` split.
- LLM judge is enabled by default. Official starter code uses Gemini 2.5 Flash with the DeepSearchQA judge prompt, but EvalScope can use any configured judge model for local runs.
- The primary metric is `f1_score`; `precision`, `recall`, and empty/invalid response rates are also reported.
- `JudgeStrategy.RULE` provides a conservative exact/substring fallback for smoke tests and is not equivalent to official LLM judging.

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `deepsearchqa` |
| **Dataset ID** | [google/deepsearchqa](https://modelscope.cn/datasets/google/deepsearchqa/summary) |
| **Paper** | [Paper](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf) |
| **Tags** | `Agent`, `Knowledge`, `QA`, `Retrieval` |
| **Metrics** | `f1_score`, `precision`, `recall` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `eval` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 900 |
| Prompt Length (Mean) | 295.54 chars |
| Prompt Length (Min/Max) | 49 / 1007 chars |

## Sample Example

*Sample example not available.*

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
    --datasets deepsearchqa \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":30}' \
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
    datasets=['deepsearchqa'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=30,
    ),
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
