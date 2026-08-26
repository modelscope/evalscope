from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict, List, Optional

from evalscope.agent.external.config import ExternalAgentConfig
from evalscope.agent.tools.python_exec import PYTHON_EXEC_TOOL_INFO
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfig, MCPServerConfigStdio
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import InferenceReturn
from evalscope.api.model import Model
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.hle.hle_adapter import SUBSET_LIST, HLEAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

_DEFAULT_MAX_STEPS = 30
_DEFAULT_TOOLS = ['python_exec']
_DEFAULT_ENVIRONMENT = 'local'

TOOL_USE_HINT = (
    '\n\nYou may use the python_exec tool to compute or verify results. '
    'If fetch or search tools are available, use them to look up information. '
    'When you are finished, give your final answer in the required format.'
)

_DESCRIPTION = """
## Overview

Humanity's Last Exam with Tools (`hle_tools`) is the tool-use counterpart of closed-book [`hle`](hle.md). It evaluates the same 2,500 expert-level questions from CAIS / Scale AI, but drives each sample through EvalScope's Native AgentLoop so the model can call code execution and optional web tools before answering.

This is a **distinct leaderboard row** from `hle`. Use `datasets=['hle']` for closed-book QA and `datasets=['hle_tools']` for the multi-turn tool-use protocol.

## Task Description

- **Task Type**: Expert-Level Question Answering with Tools (multi-turn AgentLoop)
- **Input**: Question with optional image (14% multimodal), plus `python_exec` (and optional MCP fetch/search)
- **Output**: Answer with explanation and confidence score
- **Domains**: Mathematics (41%), Physics (9%), Biology/Medicine (11%), Computer Science/AI (10%), Humanities (9%), Engineering (4%), Chemistry (7%), Other (9%)

## Key Features

- Reuses the official HLE dataset (`cais/hle` on ModelScope), judge (`GRADE: C/I`), subsets, and `include_multi_modal` extra param
- Default Native AgentLoop: `function_calling` strategy, `python_exec`, `local` environment, 30 steps
- Optional MCP `fetch` is attached automatically when `evalscope[mcp]` (and `mcp-server-fetch`) is installed — no paid search API key required
- Optional MCP web search (Brave, Tavily, …) can be added through `NativeAgentConfig.mcp_servers`
- Docker is recommended for isolated production runs but is **not** required; the default `local` environment is mock-friendly

## Evaluation Notes

- Default evaluation uses the **test** split
- Primary metric: **Accuracy** with the same HLE LLM judge
- Response format includes: Explanation, Answer, and Confidence (0-100%)
- **Note**: Set `extra_params["include_multi_modal"]` to `False` for text-only models
- Uses GRADE: C/I format for LLM judge scoring
- `datasets=['hle_tools']` enables tools by default. You do **not** need to set `TaskConfig.agent_config` unless you want to override the loop
- Passing `NativeAgentConfig` merges with defaults for any field you leave unset (`tools`, `environment`, `max_steps`, `mcp_servers`)
- `local` has no filesystem isolation; for formal runs set `environment='docker'` (see below)
- Install web fetch with `pip install evalscope[mcp]`. Paid search MCP servers are optional

## Agent Environment

Default loop (applied when `agent_config` is omitted):

- **Strategy**: `function_calling`
- **Tools**: `python_exec` (built-in). `submit` is auto-injected
- **Environment**: `local` (host subprocess). Recommended production override: `docker` + `python:3.11-slim`
- **MCP**: `mcp-server-fetch` when the optional extra is installed
- **max_steps**: 30

Override example — Docker plus fetch, and an optional search server:

```python
import sys
from evalscope import TaskConfig, run_task
from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio

run_task(TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['hle_tools'],
    dataset_args={
        'hle_tools': {
            'subset_list': ['Math'],
            'extra_params': {'include_multi_modal': False},
        }
    },
    agent_config=NativeAgentConfig(
        strategy='function_calling',
        tools=['python_exec'],
        environment='docker',
        environment_extra={'sandbox_config': {'image': 'python:3.11-slim'}},
        max_steps=30,
        mcp_servers=[
            MCPServerConfigStdio(
                command=sys.executable,
                args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                name='fetch',
            ),
            # Optional paid search, e.g. Brave / Tavily MCP — not required
        ],
    ),
    limit=10,
))
```
"""


def default_hle_tools_mcp_servers() -> List[MCPServerConfig]:
    """Return the default fetch MCP server when the optional extra is installed."""
    if importlib.util.find_spec('mcp') is None or importlib.util.find_spec('mcp_server_fetch') is None:
        return []
    return [
        MCPServerConfigStdio(
            command=sys.executable,
            args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
            name='fetch',
        )
    ]


def default_hle_tools_agent_config() -> NativeAgentConfig:
    """Return the built-in Native AgentLoop defaults for ``hle_tools``."""
    return NativeAgentConfig(
        strategy='function_calling',
        tools=list(_DEFAULT_TOOLS),
        environment=_DEFAULT_ENVIRONMENT,
        max_steps=_DEFAULT_MAX_STEPS,
        mcp_servers=default_hle_tools_mcp_servers(),
    )


def merge_hle_tools_agent_config(agent_config: Optional[NativeAgentConfig]) -> NativeAgentConfig:
    """Fill unset NativeAgentConfig fields with ``hle_tools`` defaults."""
    if agent_config is None:
        return default_hle_tools_agent_config()

    updates: Dict[str, Any] = {}
    if not agent_config.tools:
        updates['tools'] = list(_DEFAULT_TOOLS)
    if agent_config.environment is None:
        updates['environment'] = _DEFAULT_ENVIRONMENT
    if 'max_steps' not in agent_config.model_fields_set:
        updates['max_steps'] = _DEFAULT_MAX_STEPS
    if 'mcp_servers' not in agent_config.model_fields_set:
        updates['mcp_servers'] = default_hle_tools_mcp_servers()
    if not updates:
        return agent_config
    return agent_config.model_copy(update=updates)


@register_benchmark(
    BenchmarkMeta(
        name='hle_tools',
        pretty_name="Humanity's-Last-Exam-with-Tools",
        tags=[Tags.KNOWLEDGE, Tags.QA, Tags.AGENT, Tags.MULTI_TURN],
        description=_DESCRIPTION,
        dataset_id='cais/hle',
        paper_url='https://arxiv.org/abs/2501.14249',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
        evaluation_version='v1.0',
        extra_params={
            'include_multi_modal': {
                'type': 'bool',
                'description': 'Include multi-modal (image) questions during evaluation.',
                'value': True
            }
        }
    )
)
class HLEToolsAdapter(HLEAdapter):
    """HLE with a default Native AgentLoop (``python_exec`` + optional MCP fetch)."""

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        sample = super().record_to_sample(record)
        sample.tools = [PYTHON_EXEC_TOOL_INFO]
        messages = sample.input
        if isinstance(messages, list) and messages:
            system_message = messages[0]
            if getattr(system_message, 'role', None) == 'system':
                system_message.text = f'{system_message.text}{TOOL_USE_HINT}'
        return sample

    def _effective_agent_config(self) -> NativeAgentConfig:
        ac = self._task_config.agent_config if self._task_config is not None else None
        if isinstance(ac, NativeAgentConfig):
            return merge_hle_tools_agent_config(ac)
        return default_hle_tools_agent_config()

    def _on_inference(self, model: Model, sample: Sample) -> InferenceReturn:
        """Always run the Native AgentLoop, using benchmark defaults when unset.

        A TaskConfig copy is used so a shared ``agent_config`` is not mutated for
        sibling datasets in the same task (e.g. ``['hle', 'hle_tools']``).
        """
        if self._task_config is None:
            return super()._on_inference(model, sample)

        ac = self._task_config.agent_config
        if isinstance(ac, ExternalAgentConfig):
            return self._on_external_agent_inference(model, sample)

        from evalscope.agent.runner import run_native_agent

        task_config = self._task_config.model_copy(update={'agent_config': self._effective_agent_config()})
        mcp_names = [
            getattr(server, 'name', None) or getattr(server, 'command', None)
            for server in task_config.agent_config.mcp_servers
        ]
        logger.info(
            f'hle_tools agent_config: strategy={task_config.agent_config.strategy} '
            f'tools={task_config.agent_config.tools} environment={task_config.agent_config.environment} '
            f'max_steps={task_config.agent_config.max_steps} mcp_servers={mcp_names}'
        )
        return run_native_agent(
            task_config=task_config,
            model=model,
            sample=sample,
            build_sandbox_config=self.build_sandbox_config,
            extract_final_answer=self._extract_final_answer,
        )
