import json
import yaml
from typing import Any, Dict, List, Optional

from evalscope.utils import get_logger
from .tool_call import ToolCall, ToolFunction
from .tool_info import ToolInfo

logger = get_logger()


def parse_tool_call(id: str, function: str, arguments: str, tools: Optional[List[ToolInfo]] = None) -> ToolCall:
    """Parse a tool call from a JSON payload.

    Note that this function doesn't know about internal tool names so the caller
    should ammend the returned `ToolCall` by mapping the parsed `function` field from
    an internal name to an tool name and fixing up the `ToolCall` object
    as required to reflect this change.
    """
    error: Optional[str] = None
    arguments_dict: Dict[str, Any] = {}

    def report_parse_error(ex: Exception) -> None:
        nonlocal error
        error = tool_parse_error_message(arguments, ex)
        logger.info(error)

    # if the arguments is a dict, then handle it with a plain json.loads
    arguments = arguments.strip()
    if arguments.startswith('{'):
        try:
            arguments_dict = json.loads(arguments)
        except json.JSONDecodeError as ex:
            report_parse_error(ex)

    # otherwise parse it as yaml (which will pickup unquoted strings, numbers, and true/false)
    # and then create a dict that maps it to the first function argument
    elif function and tools:
        tool_info = next(
            (tool for tool in tools if tool.name == function and len(tool.parameters.properties) > 0),
            None,
        )
        if tool_info:
            param_names = list(tool_info.parameters.properties.keys())
            try:
                value = yaml.safe_load(arguments)
                arguments_dict[param_names[0]] = value
            except yaml.error.YAMLError:
                # If the yaml parser fails, we treat it as a string argument.
                arguments_dict[param_names[0]] = arguments

    # return ToolCall with error payload
    return ToolCall(
        id=id,
        function=ToolFunction(
            name=function,
            arguments=arguments_dict,
        ),
        parse_error=error,
    )


def tool_parse_error_message(arguments: str, ex: Exception) -> str:
    return f'Error parsing the following tool call arguments:\n\n{arguments}\n\nError details: {ex}'
