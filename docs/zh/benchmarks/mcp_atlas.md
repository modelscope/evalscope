# MCP-Atlas

## 概述

MCP-Atlas 是 Scale AI 推出的一项基准测试，用于评估模型在真实 Model Context Protocol (MCP) 服务器环境下的工具使用能力。该基准包含公开任务，每个任务提供提示词、允许使用的工具列表、真实工具调用轨迹（ground-truth tool trajectories），以及用于 LLM-as-judge 覆盖率评分的专家声明（expert claims）。

## 任务描述

- **任务类型**：工具使用智能体基准测试
- **输入**：自然语言任务提示词，以及每个任务对应的 MCP 工具白名单
- **输出**：在必要时调用 MCP 工具后生成的最终任务答案
- **评分方式**：由 LLM 评判最终回答是否满足每条专家定义的声明

## 核心特性

- 使用 EvalScope 原生的 AgentLoop，而非 MCP-Atlas 自带的 `mcp_eval` 补全服务。
- 直接连接 MCP-Atlas 的 `agent-environment` HTTP 服务，该服务必须暴露 `/enabled-servers`、`/list-tools` 和 `/call-tool` 端点。
- 默认根据当前启用的 MCP 服务器过滤任务，以匹配 MCP-Atlas 公开脚本在未配置全部外部 API 密钥环境下的行为。
- 仅向模型暴露任务指定的 `ENABLED_TOOLS`，避免一次性展示数百个工具。
- 在单个样本内，若对 MCP 服务器的重复调用因传输层错误失败，则提前终止后续调用。
- 报告平均 `coverage_score` 和 `pass_rate`，并支持配置通过阈值。

## 评测说明

- 运行此基准前，请先启动 MCP-Atlas 的 `agent-environment` Docker 服务，默认 URL 为 `http://localhost:1984`。
- 此 EvalScope 原生适配器旨在便于在 EvalScope 内部维护。除非单独验证当前 Scale 排行榜所用评测配置与此一致，否则不保证结果与排行榜等效。
- 要实现完整的公开数据集覆盖，需配置 MCP-Atlas 所需的外部 API 密钥和服务数据。

## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `mcp_atlas` |
| **数据集ID** | [ScaleAI/MCP-Atlas](https://modelscope.cn/datasets/ScaleAI/MCP-Atlas/summary) |
| **论文** | [Paper](https://static.scale.com/uploads/674f4cc7a74e35bcaae1c29a/MCP_Atlas.pdf) |
| **标签** | `Agent`, `MultiTurn` |
| **指标** | `coverage_score`, `pass` |
| **默认示例数** | 0-shot |
| **评测划分** | `train` |

## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 89 |
| 提示词长度（平均） | 291.67 字符 |
| 提示词长度（最小/最大） | 133 / 579 字符 |

## 样例示例

**子集**: `default`

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

## 提示模板

**提示模板：**
```text
{question}
```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `mcp_server_url` | `str` | `http://localhost:1984` | MCP-Atlas agent-environment 服务的基础 URL。 |
| `filter_enabled_servers` | `bool` | `True` | 跳过那些真实轨迹中使用了当前未启用 MCP 服务器的任务。 |
| `max_tool_calls` | `int` | `100` | 每个样本允许的最大 MCP 工具调用次数。 |
| `request_timeout` | `float` | `60.0` | MCP 工具调用的超时时间（秒）。 |
| `list_tools_timeout` | `float` | `180.0` | MCP 服务器预检和 list-tools 请求的超时时间（秒）。 |
| `use_system_prompt` | `bool` | `False` | 在每个样本前添加 MCP-Atlas 可选的系统提示。 |
| `pass_threshold` | `float` | `0.75` | 用于计算通过率的覆盖率得分阈值。 |

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets mcp_atlas \
    --agent-config '{"mode":"native","strategy":"function_calling","max_steps":100}' \
    --limit 10  # 正式评测时请删除此行
```

### 使用 Python

```python
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['mcp_atlas'],
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        max_steps=100,
    ),
    dataset_args={
        'mcp_atlas': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评测时请删除此行
)

run_task(task_cfg=task_cfg)
```
