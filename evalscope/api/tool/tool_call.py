import json
from pydantic import BaseModel, Field, JsonValue, field_validator
from typing import Any, Callable, Dict, List, Literal, Optional, Union


class ToolFunction(BaseModel):
    """Indicate that a specific tool function should be called."""

    name: str
    """The name of the tool function to call."""

    arguments: Dict[str, Any]
    """The arguments of the tool function to call"""

    @field_validator('arguments', mode='before')
    @classmethod
    def parse_arguments(cls, v):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except Exception as e:
                raise ValueError(f'arguments field string is not valid JSON: {e}')
        if not isinstance(v, dict):
            raise ValueError('arguments must be a dict or a JSON string representing a dict')
        return v


class ToolCallContent(BaseModel):
    """Content to include in tool call view."""

    title: Optional[str] = Field(default=None)
    """Optional (plain text) title for tool call content."""

    format: Literal['text', 'markdown']
    """Format (text or markdown)."""

    content: str
    """Text or markdown content."""


class ToolCallView(BaseModel):
    """Custom view of a tool call.

    Both `context` and `call` are optional. If `call` is not specified
    then the view will default to a syntax highlighted Python function call.
    """

    context: Optional[ToolCallContent] = Field(default=None)
    """Context for the tool call (i.e. current tool state)."""

    call: Optional[ToolCallContent] = Field(default=None)
    """Custom representation of tool call."""


class ToolCall(BaseModel):
    id: str
    """Unique identifier for tool call."""

    function: ToolFunction
    """Function to call."""

    internal: Optional[JsonValue] = Field(default=None)
    """Model provider specific payload - typically used to aid transformation back to model types."""

    parse_error: Optional[str] = Field(default=None)
    """Error which occurred parsing tool call."""

    view: Optional[ToolCallContent] = Field(default=None)
    """Custom view of tool call input."""

    type: Optional[str] = Field(default=None)
    """Tool call type (deprecated)."""


class ToolCallError(BaseModel):
    """Error raised by a tool call."""

    type: Literal[
        'parsing',
        'timeout',
        'unicode_decode',
        'permission',
        'file_not_found',
        'is_a_directory',
        'limit',
        'approval',
        'unknown',
    ]
    """Error type."""

    message: str
    """Error message."""


ToolChoice = Union[Literal['auto', 'any', 'none'], ToolFunction]
"""Specify which tool to call.

"auto" means the model decides; "any" means use at least one tool,
"none" means never call a tool; ToolFunction instructs the model
to call a specific function.
"""
