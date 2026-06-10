"""bash shell tool."""

from typing import Optional

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


__all__ = ['run_bash', 'BASH_TOOL_INFO']
