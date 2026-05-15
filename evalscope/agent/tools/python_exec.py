"""Python code execution tool."""

from typing import Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_agent_tool
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.api.tool.tool_info import ToolParams
from evalscope.utils.json_schema import JSONSchema

PYTHON_EXEC_TOOL_INFO = ToolInfo(
    name='python_exec',
    description=('Execute Python source code inside the sandbox environment. '
                 'Returns stdout and stderr output.'),
    parameters=ToolParams(
        properties={
            'code': JSONSchema(type='string', description='Python source code to execute.'),
            'timeout': JSONSchema(
                type='number',
                description='Maximum execution time in seconds (default: 60).',
                default=60,
            ),
        },
        required=['code'],
    ),
)


@register_agent_tool('python_exec', info=PYTHON_EXEC_TOOL_INFO)
async def run_python_exec(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
    """Execute Python code inside the sandbox environment."""
    if env is None:
        raise PermissionError("'python_exec' tool requires an AgentEnvironment.")
    args = call.function.arguments
    code: str = args.get('code', '')
    timeout: float = float(args.get('timeout', 60))
    result = await env.exec(['python3', '-c', code], timeout=timeout)
    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f'[stderr]\n{result.stderr.rstrip()}')
    if result.timed_out:
        parts.append('[TIMEOUT]')
    elif result.returncode != 0:
        parts.append(f'[exit {result.returncode}]')
    return '\n'.join(parts) if parts else '(no output)'


__all__ = ['run_python_exec', 'PYTHON_EXEC_TOOL_INFO']
