# BigCodeBench-Hard


## Overview

BigCodeBench-Hard is a curated subset of BigCodeBench containing 148 tasks that are more aligned with real-world programming tasks. These tasks require more complex reasoning and multi-step problem solving.

## Task Description

- **Task Type**: Code Generation (Python)
- **Input**: Programming task description (docstring or natural language instruction)
- **Output**: Complete Python function implementation
- **Difficulty**: Higher than BigCodeBench-Full, closer to real-world complexity

## Key Features

- 148 challenging tasks selected from BigCodeBench
- Requires complex reasoning and multi-tool usage
- Two evaluation modes: Complete (docstring) and Instruct (natural language)
- Uses unittest.TestCase for thorough correctness verification
- Supports pass@k metric calculation

## Evaluation Notes

- **Sandbox Required**: Requires sandbox environment with 70+ Python libraries pre-installed
- Two modes available via `split` parameter: `complete` (docstring completion) or `instruct` (NL instruction)
- Default timeout is 240 seconds per problem
- `calibrate` option prepends code_prompt to align function signatures
- See [sandbox documentation](https://evalscope.readthedocs.io/en/latest/user_guides/sandbox.html) for setup


## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `bigcodebench_hard` |
| **Dataset ID** | [evalscope/bigcodebench-hard](https://modelscope.cn/datasets/evalscope/bigcodebench-hard/summary) |
| **Paper** | N/A |
| **Tags** | `Coding` |
| **Metrics** | `acc` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `v0.1.4` |
| **Aggregation** | `mean_and_pass_at_k` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 148 |
| Prompt Length (Mean) | 777.94 chars |
| Prompt Length (Min/Max) | 274 / 1816 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "207adc46",
      "content": "Download all files from a specific directory on an FTP server using wget in a subprocess. Args: ftp_server (str): The FTP server address. Default is 'ftp.dlptest.com'. ftp_user (str): The FTP server username. Default is 'dlpuser'. ftp_passwor ... [TRUNCATED 798 chars] ...  FTP server.\nYou should write self-contained code starting with:\n```\nimport subprocess\nimport ftplib\nimport os\ndef task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):\n```"
    }
  ],
  "target": "    # Attempt to connect to the FTP server\n    try:\n        ftp_obj = ftplib.FTP(ftp_server)\n    except Exception as e:\n        raise Exception(f'Failed to connect to FTP server {ftp_server}: {str(e)}')\n\n    # Attempt to login to the FTP serv ... [TRUNCATED 625 chars] ...        command = f'wget ftp://{ftp_user}:{ftp_password}@{ftp_server}{ftp_dir}/{filename} -P {download_dir}'\n        subprocess.call(command, shell=True)\n        downloaded_files.append(filename)\n\n    ftp_obj.quit()\n    return downloaded_files",
  "id": 0,
  "group_id": 0,
  "metadata": {
    "task_id": "BigCodeBench/13",
    "entry_point": "task_func",
    "complete_prompt": "import subprocess\nimport ftplib\nimport os\n\ndef task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):\n    \"\"\"\n    Download all files from a specific directory on an FTP serv ... [TRUNCATED 909 chars] ... ctory. Outputs the message \"Failed to change to directory {ftp_dir} on server {ftp_server}: {str(e)}\"\n    \n    Requirements:\n    - subprocess\n    - ftplib\n    - os\n\n    Example:\n    >>> task_func()\n    ['file1.txt', 'file2.jpg', ...]\n    \"\"\"\n",
    "code_prompt": "import subprocess\nimport ftplib\nimport os\ndef task_func(ftp_server='ftp.dlptest.com', ftp_user='dlpuser', ftp_password='rNrKYTX9g7z3RgJRmxWuGHbeu', ftp_dir='/ftp/test'):\n",
    "test": "import unittest\nfrom unittest.mock import patch\nimport os\nclass TestCases(unittest.TestCase):\n    def setUp(self):\n        \"\"\"Setup a clean test environment before each test.\"\"\"\n        if not os.path.exists(\"downloaded_files\"):\n            o ... [TRUNCATED 2565 chars] ... with self.assertRaises(Exception) as context:\n            task_func(ftp_dir=\"/invalid_directory\")\n        self.assertEqual(str(context.exception), f'Failed to change to directory /invalid_directory on server ftp.dlptest.com: {error_message}')"
  }
}
```

## Prompt Template

**Prompt Template:**
```text
{prompt}
```

## Extra Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | `str` | `instruct` | Evaluation mode: "complete" (docstring completion) or "instruct" (NL instruction). Choices: ['complete', 'instruct'] |
| `version` | `str` | `default` | Dataset version. Use "default" for the latest available version. |
| `calibrate` | `bool` | `True` | Whether to prepend code_prompt to the solution for function signature alignment. |
| `docker_build_context` | `str` | `` | Optional local Docker build context. When set, overrides the default sandbox image. |
| `dockerfile` | `str` | `Dockerfile` | Dockerfile path inside docker_build_context. |
| `force_rebuild` | `bool` | `False` | Force rebuilding the optional local Docker image. |

## Sandbox Configuration

This benchmark requires a sandbox environment for code execution.

```json
{
  "image": "bigcodebench/bigcodebench-evaluate:latest",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  },
  "memory_limit": "4g"
}
```

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets bigcodebench_hard \
    --sandbox '{"enabled": true}' \
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
    datasets=['bigcodebench_hard'],
    sandbox={'enabled': True},
    dataset_args={
        'bigcodebench_hard': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


