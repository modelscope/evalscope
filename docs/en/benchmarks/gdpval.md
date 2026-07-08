# GDPval


## Overview

GDPval evaluates whether models can complete realistic economically valuable work tasks and produce requested
deliverable files. This adapter targets OpenAI's public 220-task gold subset mirrored on ModelScope as
`openai-mirror/gdpval`.

## Task Description

- **Task Type**: Agentic professional work / deliverable generation
- **Input**: A workplace-style task prompt, optionally with reference files
- **Output**: Final response text and requested files under `deliverable_files/`
- **Dataset**: OpenAI public GDPval gold subset with 220 tasks

## Key Features

- Uses the native EvalScope `AgentLoopAdapter` with bash and Python execution tools.
- Loads records and reference files from ModelScope by default.
- Mounts selected reference files read-only into the sandbox under `/reference_files`.
- Extracts files written to `deliverable_files/` before sandbox teardown.
- Generates a GDPval submission package with `deliverable_text` and `deliverable_files` columns.

## Evaluation Notes

- The default Docker image is built automatically from the bundled Dockerfile into a content-hashed local tag. Set
  `extra_params.auto_build_docker_image=false` to require a pre-built `evalscope/gdpval:latest`, or override
  `extra_params.docker_image`.
- `submission_ready` is a local readiness metric: it is 1 when the model produced final text or at least one
  deliverable file. It is not an official GDPval quality score.
- EvalScope does not run a local GDPval judge. Use the exported submission package with OpenAI's official GDPval judge
  to obtain quality scores.
- Full document/spreadsheet/slide quality depends on the GDPval runtime image. Thin Python images are useful only for
  plumbing smoke tests.

## Scoring and Submission

- EvalScope writes a local submission folder under the reports directory.
- The submission contains `deliverable_text` and `deliverable_files` fields in the GDPval dataset format.
- Official GDPval grading is external. Run OpenAI's official GDPval judge on the exported submission package.


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `gdpval` |
| **Dataset ID** | [openai-mirror/gdpval](https://modelscope.cn/datasets/openai-mirror/gdpval/summary) |
| **Paper** | N/A |
| **Tags** | `Agent`, `Knowledge`, `MultiTurn` |
| **Metrics** | `submission_ready` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 220 |
| Prompt Length (Mean) | 2742.59 chars |
| Prompt Length (Min/Max) | 1058 / 7160 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "3315415d",
      "content": "You are an auditor and as part of an audit engagement, you are tasked with reviewing and testing the accuracy of reported Anti-Financial Crime Risk Metrics.\n\nThe attached spreadsheet titled ‘Population’ contains Anti-Financial Crime Risk Metr ... [TRUNCATED 2069 chars] ...  folder named `deliverable_files` in the sandbox working directory.\nWe will grade your final message as part of the deliverable, but requested documents, spreadsheets, slides, media,\nor archives should be actual files in `deliverable_files`.\n"
    }
  ],
  "target": "",
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
    "task_id": "83d10b06-26d1-4636-a32c-23f92c57f30b",
    "sector": "Professional, Scientific, and Technical Services",
    "occupation": "Accountants and Auditors",
    "prompt": "You are an auditor and as part of an audit engagement, you are tasked with reviewing and testing the accuracy of reported Anti-Financial Crime Risk Metrics.\n\nThe attached spreadsheet titled ‘Population’ contains Anti-Financial Crime Risk Metr ... [TRUNCATED 1526 chars] ... across all Divisions and sub-Divisions.\n\n4. Create a new spreadsheet titled ‘Sample’:\n- Tab 1: Selected sample, copied from the original ‘Population’ sheet, with selected rows marked in column K.\n- Tab 2: Workings for sample size calculation.",
    "reference_files": [
      "reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population v2.xlsx"
    ],
    "reference_file_urls": [
      "https://huggingface.co/datasets/openai/gdpval/resolve/main/reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population%20v2.xlsx"
    ],
    "reference_file_hf_uris": [
      "hf://datasets/openai/gdpval@main/reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population%20v2.xlsx"
    ],
    "reference_paths": [
      "reference_files/Population v2.xlsx"
    ],
    "sandbox_reference_paths": [
      "/reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population v2.xlsx"
    ],
    "rubric_pretty": "[+2] The submitted deliverable is an Excel workbook file whose basename is 'Sample' (accept .xlsx, .xls, or .xlsm).\n\n[+2] The workbook contains a worksheet named exactly 'Sample Size Calculation' (case-insensitive, ignoring surrounding spaces ... [TRUNCATED 4861 chars] ... ntage changes (e.g., |J| ≥ 100%) are made easily identifiable (such as by a separate flag, note, or conditional formatting).\n\n[+1] The first worksheet is named 'Sample' (case-insensitive).\n\n[+5] Overall formatting and style of the deliverable",
    "rubric_json": "[{\"score\": 2, \"criterion\": \"The submitted deliverable is an Excel workbook file whose basename is 'Sample' (accept .xlsx, .xls, or .xlsm).\", \"required\": null, \"rubric_item_id\": \"1d43f1eb-4011-47ac-8ad7-a3c467639a6a\", \"author_type\": \"human\", \" ... [TRUNCATED 11817 chars] ... ull}, {\"score\": 5, \"criterion\": \"Overall formatting and style of the deliverable\", \"required\": null, \"rubric_item_id\": \"a64588ed-db04-4b8b-b3b8-3674ddcf10d1\", \"author_type\": \"human\", \"tags\": [\"true\"], \"read_only\": null, \"form_content\": null}]",
    "dataset_id": "openai-mirror/gdpval",
    "dataset_hub": "modelscope"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{question}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_steps` | `int` | `250` | Maximum number of agent steps per sample. |
| `command_timeout` | `float` | `180.0` | Default per-command timeout in seconds. |
| `docker_image` | `str` | `evalscope/gdpval:latest` | Docker image used as the per-sample sandbox. |
| `auto_build_docker_image` | `bool` | `True` | Automatically build the default GDPval Docker image if it is missing locally. |
| `network_enabled` | `bool` | `True` | Allow the sandbox to access the network. |
| `download_reference_files` | `bool` | `True` | Download each selected sample reference file from the dataset hub before inference. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets gdpval \
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
    datasets=['gdpval'],
    dataset_args={
        'gdpval': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```
