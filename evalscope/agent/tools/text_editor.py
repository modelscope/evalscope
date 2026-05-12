"""File system tools: read_file and write_file."""

from typing import Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.registry import register_agent_tool
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.api.tool.tool_info import ToolParams
from evalscope.utils.json_schema import JSONSchema

READ_FILE_TOOL_INFO = ToolInfo(
    name='read_file',
    description='Read the UTF-8 text content of a file inside the environment.',
    parameters=ToolParams(
        properties={
            'path': JSONSchema(
                type='string',
                description='Absolute or relative path to the file to read.',
            ),
        },
        required=['path'],
    ),
)

WRITE_FILE_TOOL_INFO = ToolInfo(
    name='write_file',
    description='Write UTF-8 text content to a file inside the environment.',
    parameters=ToolParams(
        properties={
            'path': JSONSchema(
                type='string',
                description='Path of the file to write (created if it does not exist).',
            ),
            'content': JSONSchema(
                type='string',
                description='UTF-8 text content to write.',
            ),
        },
        required=['path', 'content'],
    ),
)


@register_agent_tool('read_file', info=READ_FILE_TOOL_INFO)
async def run_read_file(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
    """Read a file from inside the environment."""
    if env is None:
        raise PermissionError("'read_file' tool requires an AgentEnvironment.")
    path: str = call.function.arguments.get('path', '')
    return await env.read_file(path)


@register_agent_tool('write_file', info=WRITE_FILE_TOOL_INFO)
async def run_write_file(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
    """Write a file inside the environment."""
    if env is None:
        raise PermissionError("'write_file' tool requires an AgentEnvironment.")
    path: str = call.function.arguments.get('path', '')
    content: str = call.function.arguments.get('content', '')
    await env.write_file(path, content)
    return f"File '{path}' written ({len(content)} chars)."


__all__ = ['run_read_file', 'run_write_file', 'READ_FILE_TOOL_INFO', 'WRITE_FILE_TOOL_INFO']
