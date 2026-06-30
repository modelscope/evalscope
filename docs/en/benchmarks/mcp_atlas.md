# MCP-Atlas

## Overview

MCP-Atlas is a Scale AI benchmark for evaluating tool-use competency with real Model Context Protocol
(MCP) servers. It contains public tasks with prompts, allowed tool lists, ground-truth tool trajectories,
and expert claims used for LLM-as-judge coverage scoring.

## Task Description

- **Task Type**: Tool-use agent benchmark
- **Input**: Natural-language task prompt plus a per-task MCP tool allowlist
- **Output**: Final task answer generated after using MCP tools when useful
- **Grading**: LLM judge checks whether the final response fulfills each expert-defined claim

## Key Features

- Uses EvalScope's native AgentLoop rather than MCP-Atlas's `mcp_eval` completion service.
- Connects directly to the MCP-Atlas `agent-environment` HTTP service, which must expose `/enabled-servers`,
  `/list-tools`, and `/call-tool`.
- Filters tasks by currently enabled MCP servers by default, matching MCP-Atlas's public-script behavior for
  environments without every external API key configured.
- Exposes only the task's `ENABLED_TOOLS` to the model to avoid advertising hundreds of tools at once.
- Short-circuits repeated calls to MCP servers that hit transport-level failures inside the same sample.
- Reports mean `coverage_score` and `pass_rate` with a configurable pass threshold.

## Evaluation Notes

- Start the MCP-Atlas `agent-environment` Docker service before running this benchmark. The default URL is
  `http://localhost:1984`.
- This EvalScope-native adapter is intended to be maintainable inside EvalScope. It is not claimed to be
  leaderboard-equivalent unless the current Scale leaderboard harness settings are separately verified.
- Full public-set coverage requires configuring the external API keys and service data required by MCP-Atlas.

## Properties

| Property | Value |
|----------|-------|
| **Benchmark Name** | `mcp_atlas` |
| **Dataset ID** | [ScaleAI/MCP-Atlas](https://modelscope.cn/datasets/ScaleAI/MCP-Atlas/summary) |
| **Paper** | [Paper](https://static.scale.com/uploads/674f4cc7a74e35bcaae1c29a/MCP_Atlas.pdf) |
| **Tags** | `Agent`, `MultiTurn` |
| **Metrics** | `coverage_score`, `pass` |
| **Default Shots** | 0-shot |
| **Evaluation Split** | `train` |


## Data Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 89 |
| Prompt Length (Mean) | 291.67 chars |
| Prompt Length (Min/Max) | 133 / 579 chars |

## Sample Example

**Subset**: `default`

```json
{
  "input": [
    {
      "id": "5dc7f8ae",
      "content": "I've been working on a local project for telling stories where I've already built a few react components. I want to check the required dependency version for the component that has the fewest lines of code."
    }
  ],
  "target": "[\"['The component with the fewest lines of code is AspectRatio', 'The AspectRatio component uses `@radix-ui/react-aspect-ratio`', 'The required dependency version for `@radix-ui/react-aspect-ratio` is\\\\n \\\"^1.0.3\\\"']\"]",
  "id": 0,
  "group_id": 0,
  "tools": [
    {
      "name": "filesystem_read_multiple_files",
      "description": "Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.",
      "parameters": {
        "properties": {
          "paths": {
            "type": "array",
            "description": "Array of file paths to read. Each path must be a string pointing to a valid file within allowed directories.",
            "items": {
              "type": "string"
            }
          }
        },
        "required": [
          "paths"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "filesystem_directory_tree",
      "description": "Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.",
      "parameters": {
        "properties": {
          "path": {
            "type": "string"
          },
          "excludePatterns": {
            "type": "array",
            "default": [],
            "items": {
              "type": "string"
            }
          }
        },
        "required": [
          "path"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "filesystem_list_allowed_directories",
      "description": "Returns the list of directories that this server is allowed to access. Subdirectories within these allowed directories are also accessible. Use this to understand which directories and their nested paths are available before trying to access files.",
      "parameters": {
        "properties": {},
        "required": [],
        "additionalProperties": false
      }
    },
    {
      "name": "git_git_status",
      "description": "Shows the working tree status",
      "parameters": {
        "properties": {
          "repo_path": {
            "type": "string"
          }
        },
        "required": [
          "repo_path"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "git_git_diff_unstaged",
      "description": "Shows changes in the working directory that are not yet staged",
      "parameters": {
        "properties": {
          "repo_path": {
            "type": "string"
          },
          "context_lines": {
            "type": "integer",
            "default": 3
          }
        },
        "required": [
          "repo_path"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "git_git_diff_staged",
      "description": "Shows changes that are staged for commit",
      "parameters": {
        "properties": {
          "repo_path": {
            "type": "string"
          },
          "context_lines": {
            "type": "integer",
            "default": 3
          }
        },
        "required": [
          "repo_path"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "git_git_diff",
      "description": "Shows differences between branches or commits",
      "parameters": {
        "properties": {
          "repo_path": {
            "type": "string"
          },
          "target": {
            "type": "string"
          },
          "context_lines": {
            "type": "integer",
            "default": 3
          }
        },
        "required": [
          "repo_path",
          "target"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "git_git_log",
      "description": "Shows the commit logs",
      "parameters": {
        "properties": {
          "repo_path": {
            "type": "string"
          },
          "max_count": {
            "type": "integer",
            "default": 10
          }
        },
        "required": [
          "repo_path"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "git_git_show",
      "description": "Shows the contents of a commit",
      "parameters": {
        "properties": {
          "repo_path": {
            "type": "string"
          },
          "revision": {
            "type": "string"
          }
        },
        "required": [
          "repo_path",
          "revision"
        ],
        "additionalProperties": false
      }
    },
    {
      "name": "fetch_fetch",
      "description": "Fetches a URL from the internet and optionally extracts its contents as markdown.\n\nAlthough originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.",
      "parameters": {
        "properties": {
          "url": {
            "type": "string",
            "format": "uri",
            "description": "URL to fetch"
          },
          "max_length": {
            "type": "integer",
            "description": "Maximum number of characters to return.",
            "default": 5000
          },
          "start_index": {
            "type": "integer",
            "description": "On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
            "default": 0
          },
          "raw": {
            "type": "boolean",
            "description": "Get the actual HTML content of the requested page, without simplification.",
            "default": false
          }
        },
        "required": [
          "url"
        ],
        "additionalProperties": false
      }
    },
    "... [TRUNCATED 7 more items] ..."
  ],
  "metadata": {
    "task_id": "689e0b1d9c8e2ac413c1f25c",
    "prompt": "I've been working on a local project for telling stories where I've already built a few react components. I want to check the required dependency version for the component that has the fewest lines of code.",
    "enabled_tools": [
      "filesystem_read_multiple_files",
      "filesystem_directory_tree",
      "filesystem_list_allowed_directories",
      "git_git_status",
      "git_git_diff_unstaged",
      "git_git_diff_staged",
      "git_git_diff",
      "git_git_log",
      "git_git_show",
      "fetch_fetch",
      "... [TRUNCATED 7 more items] ..."
    ],
    "trajectory": "[{\"content\":\"First, I'll use `cli-mcp-server_show_security_rules` to check the security rules.\",\"role\":\"assistant\",\"tool_calls\":[{\"function\":{\"arguments\":\"{}\",\"name\":\"cli-mcp-server_show_security_rules\"},\"id\":\"0035cc9a-6276-4b69-9b4f-bcb096a7 ... [TRUNCATED 14275 chars] ... n  \\\"devDependencies\\\": {\\n    \\\"dotenv\\\": \\\"16.0.3\\\",\\n    \\\"tsx\\\": \\\"^3.12.8\\\"\\n  }\\n}\\n\",\"type\":\"text\"},{\"text\":\"\\nCommand completed with return code: 0\",\"type\":\"text\"}],\"role\":\"tool\",\"tool_call_id\":\"6fb02c16-98f1-4a8d-843b-93d3f3f4d02c\"}]",
    "gtfa_claims": [
      "['The component with the fewest lines of code is AspectRatio', 'The AspectRatio component uses `@radix-ui/react-aspect-ratio`', 'The required dependency version for `@radix-ui/react-aspect-ratio` is\\n \"^1.0.3\"']"
    ],
    "required_servers": [
      "cli-mcp-server",
      "filesystem"
    ],
    "mcp_server_url": "http://localhost:1984"
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
| `mcp_server_url` | `str` | `http://localhost:1984` | MCP-Atlas agent-environment base URL. |
| `filter_enabled_servers` | `bool` | `True` | Skip tasks whose ground-truth trajectory uses MCP servers that are not currently enabled. |
| `max_steps` | `int` | `100` | Maximum number of EvalScope agent loop steps per sample. |
| `max_tool_calls` | `int` | `100` | Maximum MCP tool calls allowed per sample. |
| `request_timeout` | `float` | `60.0` | Timeout in seconds for MCP tool calls. |
| `list_tools_timeout` | `float` | `180.0` | Timeout in seconds for MCP server preflight and list-tools requests. |
| `use_system_prompt` | `bool` | `False` | Prepend the MCP-Atlas optional system prompt to every sample. |
| `pass_threshold` | `float` | `0.75` | Coverage score threshold used to compute pass rate. |

## Usage

### Using CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mcp_atlas \
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
    datasets=['mcp_atlas'],
    dataset_args={
        'mcp_atlas': {
            # extra_params: {}  # uses default extra parameters
        }
    },
    limit=10,  # Remove this line for formal evaluation
)

run_task(task_cfg=task_cfg)
```


