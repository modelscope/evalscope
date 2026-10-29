import json
from typing import Any, Dict, List, Optional

import yaml
from jsonschema import SchemaError, ValidationError, validate

from evalscope.utils import get_logger

from .tool_call import ToolCall, ToolFunction
from .tool_info import ToolInfo

logger = get_logger()


def parse_tool_call(id: str, function: str, arguments: str, tools: Optional[List[ToolInfo]] = None) -> ToolCall:
    """Parse a tool call from a JSON payload.

    Note that this function doesn't know about internal tool names so the caller
    should amend the returned `ToolCall` by mapping the parsed `function` field from
    an internal name to a tool name and fixing up the `ToolCall` object
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


def validate_tool_arguments(call: ToolCall, tool: ToolInfo) -> Optional[str]:
    """Check ``call.function.arguments`` against the schema advertised in ``tool``.

    Returns ``None`` when the arguments satisfy ``tool.parameters``, otherwise a
    short description of the first violation, for example
    ``"'query' is a required property"`` or
    ``"'abc' is not of type 'integer' (at 'limit')"``.

    The schema is ``tool.parameters.model_dump(exclude_none=True)``: exactly
    the JSON Schema the model was shown, so a call is judged against the
    contract it was given rather than against a handler's private
    expectations. A schema jsonschema cannot compile is treated as
    unconstrained, so a broken declaration never blocks a well-formed call.
    """
    schema = tool.parameters.model_dump(exclude_none=True)
    try:
        validate(instance=call.function.arguments, schema=schema)
    except ValidationError as exc:
        path = '/'.join(str(part) for part in exc.absolute_path)
        return f'{exc.message} (at {path!r})' if path else exc.message
    except SchemaError as exc:
        logger.debug(f'validate_tool_arguments: schema of tool {tool.name!r} is invalid, skipping: {exc.message}')
        return None
    return None
