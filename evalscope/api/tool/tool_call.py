from dataclasses import dataclass, field
from pydantic import BaseModel, Field, JsonValue
from typing import Any, Callable, Dict, List, Literal, Optional, Union


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


@dataclass
class ToolCall:
    id: str
    """Unique identifier for tool call."""

    function: str
    """Function called."""

    arguments: Dict[str, Any]
    """Arguments to function."""

    internal: Optional[JsonValue] = field(default=None)
    """Model provider specific payload - typically used to aid transformation back to model types."""

    parse_error: Optional[str] = field(default=None)
    """Error which occurred parsing tool call."""

    view: Optional[ToolCallContent] = field(default=None)
    """Custom view of tool call input."""

    type: Optional[str] = field(default=None)
    """Tool call type (deprecated)."""


@dataclass
class ToolCallError:
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
        # Retained for backward compatibility when loading logs created with an older
        # version of inspect.
        'output_limit',
    ]
    """Error type."""

    message: str
    """Error message."""


@dataclass
class ToolFunction:
    """Indicate that a specific tool function should be called."""

    name: str
    """The name of the tool function to call."""


ToolChoice = Union[Literal['auto', 'any', 'none'], ToolFunction]
"""Specify which tool to call.

"auto" means the model decides; "any" means use at least one tool,
"none" means never call a tool; ToolFunction instructs the model
to call a specific function.
"""
