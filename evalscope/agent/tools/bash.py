"""bash shell tool."""

from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.agent import AgentEnvironment
from evalscope.api.agent.types import ExecResult
from evalscope.api.registry import register_agent_tool
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.api.tool.tool_info import ToolParams
from evalscope.utils.json_schema import JSONSchema

BASH_TOOL_INFO = ToolInfo(
    name='bash',
    description=(
        'Execute a bash command inside the sandbox environment. '
        'Returns the combined stdout / stderr output of the command.'
    ),
    parameters=ToolParams(
        properties={
            'command': JSONSchema(type='string', description='The bash command to execute.'),
            'timeout': JSONSchema(
                type='number',
                description='Maximum execution time in seconds (default: 60).',
                default=60,
            ),
        },
        required=['command'],
    ),
)


@register_agent_tool('bash', info=BASH_TOOL_INFO)
async def run_bash(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
    """Execute a bash command in the sandbox environment."""
    if env is None:
        raise PermissionError("'bash' tool requires an AgentEnvironment.")
    args = call.function.arguments
    command: str = args.get('command', '')
    timeout: float = float(args.get('timeout', 60))
    result = await env.exec(['/bin/bash', '-c', command], timeout=timeout)
    return _format_exec_result(result)


def apply_bash_command_timeout_defaults(
    handlers: Dict[str, Any],
    tools: List[ToolInfo],
    command_timeout: Optional[float],
) -> Tuple[Dict[str, Any], List[ToolInfo]]:
    """Apply a native runtime default timeout to bash calls and schema."""
    if command_timeout is None:
        return handlers, tools
    updated_handlers = _apply_bash_command_timeout(handlers, command_timeout)
    updated_tools = _apply_bash_tool_timeout_default(tools, command_timeout)
    return updated_handlers, updated_tools


def _apply_bash_command_timeout(handlers: Dict[str, Any], command_timeout: float) -> Dict[str, Any]:
    if 'bash' not in handlers:
        return handlers

    bash_handler = handlers['bash']

    async def run_bash_with_default_timeout(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
        args = dict(call.function.arguments or {})
        if 'timeout' not in args:
            call = call.model_copy(
                update={
                    'function': call.function.model_copy(update={'arguments': {
                        **args,
                        'timeout': command_timeout,
                    }})
                }
            )
        return await bash_handler(call, env)

    return {**handlers, 'bash': run_bash_with_default_timeout}


def _apply_bash_tool_timeout_default(tools: List[ToolInfo], command_timeout: float) -> List[ToolInfo]:
    updated_tools: List[ToolInfo] = []
    for tool in tools:
        if tool.name != 'bash':
            updated_tools.append(tool)
            continue
        copied = tool.model_copy(deep=True)
        timeout_param = copied.parameters.properties.get('timeout')
        if timeout_param is not None:
            timeout_param.default = command_timeout
            timeout_param.description = f'Maximum execution time in seconds (default: {command_timeout:g}).'
        updated_tools.append(copied)
    return updated_tools


def _format_exec_result(result: ExecResult) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f'[stderr]\n{result.stderr.rstrip()}')
    if result.timed_out:
        parts.append('[TIMEOUT]')
    elif result.returncode != 0:
        parts.append(f'[exit {result.returncode}]')
    return '\n'.join(parts) if parts else '(no output)'


__all__ = ['run_bash', 'BASH_TOOL_INFO', 'apply_bash_command_timeout_defaults']
