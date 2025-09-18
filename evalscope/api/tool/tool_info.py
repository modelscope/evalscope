import inspect
from dataclasses import dataclass
from docstring_parser import Docstring, parse
from pydantic import BaseModel, Field, field_validator
from typing import Any, Callable, Dict, List, Literal, Optional, TypeAlias, Union, get_args, get_type_hints

from evalscope.utils.json_schema import JSONSchema, JSONType, json_schema, python_type_to_json_type

ToolParam: TypeAlias = JSONSchema
"""Description of tool parameter in JSON Schema format."""


class Tool:

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ...


class ToolParams(BaseModel):
    """Description of tool parameters object in JSON Schema format."""

    type: Literal['object'] = Field(default='object')
    """Params type (always 'object')"""

    properties: Dict[str, ToolParam] = Field(default_factory=dict)
    """Tool function parameters."""

    required: List[str] = Field(default_factory=list)
    """List of required fields."""

    additionalProperties: bool = Field(default=False)
    """Are additional object properties allowed? (always `False`)"""


@dataclass
class ToolDescription:
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[ToolParams] = None


def tool_description(tool: Tool) -> ToolDescription:
    return getattr(tool, TOOL_DESCRIPTION, ToolDescription())


def set_tool_description(tool: Tool, description: ToolDescription) -> None:
    setattr(tool, TOOL_DESCRIPTION, description)


TOOL_DESCRIPTION = '__TOOL_DESCRIPTION__'


class ToolInfo(BaseModel):
    """Specification of a tool (JSON Schema compatible)

    If you are implementing a ModelAPI, most LLM libraries can
    be passed this object (dumped to a dict) directly as a function
    specification. For example, in the OpenAI provider:

    ```python
    ChatCompletionToolParam(
        type="function",
        function=tool.model_dump(exclude_none=True),
    )
    ```

    In some cases the field names don't match up exactly. In that case
    call `model_dump()` on the `parameters` field. For example, in the
    Anthropic provider:

    ```python
    ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=tool.parameters.model_dump(exclude_none=True),
    )
    ```
    """

    name: str
    """Name of tool."""
    description: str
    """Short description of tool."""
    parameters: ToolParams = Field(default_factory=ToolParams)
    """JSON Schema of tool parameters object."""
    options: Optional[Dict[str, object]] = Field(default=None)
    """Optional property bag that can be used by the model provider to customize the implementation of the tool"""


def parse_tool_info(func: Callable[..., Any]) -> ToolInfo:
    # tool may already have registry attributes w/ tool info
    description = tool_description(func)
    if (description.name and description.description and description.parameters is not None):
        return ToolInfo(
            name=description.name,
            description=description.description,
            parameters=description.parameters,
        )

    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    docstring = inspect.getdoc(func)
    parsed_docstring: Optional[Docstring] = parse(docstring) if docstring else None

    info = ToolInfo(name=func.__name__, description='')

    for param_name, param in signature.parameters.items():
        tool_param = ToolParam()

        # Parse docstring
        docstring_info = parse_docstring(docstring, param_name)

        # Get type information from type annotations
        if param_name in type_hints:
            tool_param = json_schema(type_hints[param_name])
        # as a fallback try to parse it from the docstring
        # (this is minimally necessary for backwards compatiblity
        #  with tools gen1 type parsing, which only used docstrings)
        elif 'docstring_type' in docstring_info:
            json_type = python_type_to_json_type(docstring_info['docstring_type'])
            if json_type and (json_type in get_args(JSONType)):
                tool_param = ToolParam(type=json_type)

        # Get default value
        if param.default is param.empty:
            info.parameters.required.append(param_name)
        else:
            tool_param.default = param.default

        # Add description from docstring
        if 'description' in docstring_info:
            tool_param.description = docstring_info['description']

        # append the tool param
        info.parameters.properties[param_name] = tool_param

    # Add function description if available
    if parsed_docstring:
        if parsed_docstring.description:
            info.description = parsed_docstring.description.strip()
        elif parsed_docstring.long_description:
            info.description = parsed_docstring.long_description.strip()
        elif parsed_docstring.short_description:
            info.description = parsed_docstring.short_description.strip()

        # Add examples if available
        if parsed_docstring.examples:
            examples = '\n\n'.join([(example.description or '') for example in parsed_docstring.examples])
            info.description = f'{info.description}\n\nExamples\n\n{examples}'

    return info


def parse_docstring(docstring: Optional[str], param_name: str) -> Dict[str, str]:
    if not docstring:
        return {}

    parsed_docstring: Docstring = parse(docstring)

    for param in parsed_docstring.params:
        if param.arg_name == param_name:
            schema: Dict[str, str] = {'description': param.description or ''}

            if param.type_name:
                schema['docstring_type'] = param.type_name

            return schema

    return {}
