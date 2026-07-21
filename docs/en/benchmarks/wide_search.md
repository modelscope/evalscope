# WideSearch


## Overview

WideSearch evaluates search agents on broad web information-seeking tasks. Each task asks the agent to collect many
atomic facts and return one structured Markdown table. EvalScope uses the ModelScope
`bytedance-community/WideSearch` dataset.

## Task Description

- **Task Type**: Multi-turn search agent
- **Input**: Natural-language collection request with an explicit table schema
- **Output**: Complete Markdown table
- **Dataset**: 200 tasks in the ``full`` split; 100 English and 100 Chinese

## Key Features

- Official single-agent protocol: language-specific system prompt, ``function_calling``, and 50 default steps.
- Bash is available by default in a per-sample temporary local directory; Docker sandbox and MCP servers are optional.
- A single full run derives ``all``, ``en``, and ``zh`` reports without repeated inference.

## Evaluation Notes

- Uses the official Markdown table alignment and hybrid rule/LLM scoring semantics.
- Requires ``judge_strategy='auto'`` or ``'llm'`` with explicit ``judge_model_args``; rule-only scoring is unsupported.
- See the [WideSearch usage guide](https://evalscope.readthedocs.io/en/latest/third_party/wide_search.html) for runtime
  examples and paper-style repeat settings.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `wide_search` |
| **Dataset ID** | [bytedance-community/WideSearch](https://modelscope.cn/datasets/bytedance-community/WideSearch/summary) |
| **Paper** | [Paper](https://arxiv.org/abs/2508.07999) |
| **Tags** | `Agent`, `MultiTurn`, `Retrieval` |
| **Metrics** | `success_rate`, `row_precision`, `row_recall`, `row_f1`, `item_precision`, `item_recall`, `item_f1` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `full` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 200 |
| Prompt Length (Mean) | 631.01 chars |
| Prompt Length (Min/Max) | 182 / 1807 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "59f736f6",
      "content": "My son is about to start his university applications in 2025 for postgraduates but he’s still uncertain about both his major and which universities to apply to. Could you help me find the top five universities in each of the five broad subjec ... [TRUNCATED 691 chars] ... names in English. \nUse only Arabic numerals in the ranking, for example: 1.\nDon't ask me any questions, just output the results according to the columns without omitting cells arbitrarily. The output format is \n```markdown\n{data_content}\n```."
    }
  ],
  "target": "Subject,University,Country,QS World University Rankings by Subject 2025,QS World University Rankings 2025,Times Higher Education  World University Rankings 2025,Home Page,Application Deadline,Application Fee\nArts & Humanities,Harvard Universi ... [TRUNCATED 2837 chars] ... nuary 5,$90\nSocial Sciences & Management,Massachusetts Institute of Technology,United States,4,1,2,https://www.mit.edu/,January 6 ,$75\nSocial Sciences & Management,University of Cambridge,United Kingdom,5,5,5,https://www.cam.ac.uk/,Oct 15,£60",
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
    "instance_id": "ws_en_001",
    "language": "en",
    "evaluation": {
      "unique_columns": [
        "subject",
        "university"
      ],
      "required": [
        "subject",
        "university",
        "qsworlduniversityrankingsbysubject2025",
        "qsworlduniversityrankings2025",
        "timeshighereducationworlduniversityrankings2025",
        "homepage",
        "applicationdeadline",
        "applicationfee"
      ],
      "eval_pipeline": {
        "applicationdeadline": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nThe month and day must be correct"
        },
        "applicationfee": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "llm_judge"
          ],
          "criterion": "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence.\nIf there are multiple fees in the reference answer, all must be included."
        },
        "homepage": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "url_match"
          ]
        },
        "subject": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "university": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankingsbysubject2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "qsworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        },
        "timeshighereducationworlduniversityrankings2025": {
          "preprocess": [
            "norm_str"
          ],
          "metric": [
            "exact_match"
          ]
        }
      }
    }
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
    --datasets wide_search \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":50}' \
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
    datasets=['wide_search'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=50,
    ),
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
