from __future__ import annotations

from typing import Any, Dict

from evalscope.constants import HubType

DATASET_ID = 'ScaleAI/MCP-Atlas'
DEFAULT_MCP_SERVER_URL = 'http://localhost:1984'
DEFAULT_SYSTEM_PROMPT = (
    'Role: You are a factual, tool-aware assistant connected to a variety of tools. '
    'Use the available tools to answer the user query. Do not ask the user for clarification; '
    'fully complete the task using the information provided in the prompt.'
)

EXTRA_PARAMS: Dict[str, Any] = {
    'dataset_hub': {
        'type': 'str',
        'description': 'Dataset hub used to load MCP-Atlas records.',
        'value': HubType.MODELSCOPE,
        'choices': [HubType.MODELSCOPE, HubType.LOCAL],
    },
    'dataset_revision': {
        'type': 'str',
        'description': 'Optional dataset revision. Empty uses the hub default.',
        'value': '',
    },
    'local_path': {
        'type': 'str',
        'description': 'Optional local MCP-Atlas CSV file path. Overrides hub loading when set.',
        'value': '',
    },
    'mcp_server_url': {
        'type': 'str',
        'description': 'MCP-Atlas agent-environment base URL.',
        'value': DEFAULT_MCP_SERVER_URL,
    },
    'filter_enabled_servers': {
        'type': 'bool',
        'description': 'Skip tasks whose ground-truth trajectory uses MCP servers that are not currently enabled.',
        'value': True,
    },
    'max_steps': {
        'type': 'int',
        'description': 'Maximum number of EvalScope agent loop steps per sample.',
        'value': 100,
    },
    'max_tool_calls': {
        'type': 'int',
        'description': 'Maximum MCP tool calls allowed per sample.',
        'value': 100,
    },
    'request_timeout': {
        'type': 'float',
        'description': 'Timeout in seconds for MCP tool calls.',
        'value': 60.0,
    },
    'list_tools_timeout': {
        'type': 'float',
        'description': 'Timeout in seconds for MCP server preflight and list-tools requests.',
        'value': 180.0,
    },
    'use_system_prompt': {
        'type': 'bool',
        'description': 'Prepend the MCP-Atlas optional system prompt to every sample.',
        'value': False,
    },
    'pass_threshold': {
        'type': 'float',
        'description': 'Coverage score threshold used to compute pass rate.',
        'value': 0.75,
    },
}

DESCRIPTION = """
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
""".strip()
