import uuid
from pydantic import BaseModel, Field, JsonValue, model_validator
from typing import Any, Dict, List, Literal, Optional, Type, Union

from evalscope.api.tool import ToolCall, ToolCallError
from .content import Content, ContentReasoning, ContentText
from .utils import parse_content_with_reasoning


class ChatMessageBase(BaseModel):
    """Base class for chat messages."""

    id: Optional[str] = Field(default=None)
    """Unique identifer for message."""

    content: Union[str, List[Content]]
    """Content (simple string or list of content objects)"""

    source: Optional[Literal['input', 'generate']] = Field(default=None)
    """Source of message."""

    metadata: Optional[Dict[str, Any]] = Field(default=None)
    """Additional message metadata."""

    internal: Optional[JsonValue] = Field(default=None)
    """Model provider specific payload - typically used to aid transformation back to model types."""

    def model_post_init(self, __context: Any) -> None:
        # Generate ID
        if self.id is None:
            self.id = uuid.uuid4().hex[:8]  # Shorten to 8 characters for simplicity

    @property
    def text(self) -> str:
        """Get the text content of this message.

        ChatMessage content is very general and can contain either
        a simple text value or a list of content parts (each of which
        can either be text or an image). Solvers (e.g. for prompt
        engineering) often need to interact with chat messages with
        the assumption that they are a simple string. The text
        property returns either the plain str content, or if the
        content is a list of text and images, the text items
        concatenated together (separated by newline)
        """
        if isinstance(self.content, str):
            return self.content
        else:
            all_text = [content.text for content in self.content if content.type == 'text']
            return '\n'.join(all_text)

    @text.setter
    def text(self, text: str) -> None:
        """Set the primary text content for this message.

        ChatMessage content is very general and can contain either
        a simple text value or a list of content parts (each of which
        can either be text or an image). Solvers (e.g. for prompt
        engineering) often need to interact with chat messages with
        the assumption that they are a simple string. The text property
        sets text either to content directly (if it is a `str`) or to
        the first text content item in the message (inserting one at
        the beginning if necessary). If there are multiple text content
        items in the message then after the set there will be only
        one remaining (image content will remain).
        """
        if isinstance(self.content, str):
            self.content = text
        else:
            all_other = [content for content in self.content if content.type != 'text']
            self.content = all_other + [ContentText(text=text)]


class ChatMessageSystem(ChatMessageBase):
    """System chat message."""

    role: Literal['system'] = Field(default='system')
    """Conversation role."""


class ChatMessageUser(ChatMessageBase):
    """User chat message."""

    role: Literal['user'] = Field(default='user')
    """Conversation role."""

    tool_call_id: Optional[List[str]] = Field(default=None)
    """ID(s) of tool call(s) this message has the content payload for."""


class ChatMessageAssistant(ChatMessageBase):
    """Assistant chat message."""

    role: Literal['assistant'] = Field(default='assistant')
    """Conversation role."""

    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    """Tool calls made by the model."""

    model: Optional[str] = Field(default=None)
    """Model used to generate assistant message."""

    # Some OpenAI compatible REST endpoints include reasoning as a field alongside
    # content, however since this field doesn't exist in the OpenAI interface,
    # hosting providers (so far we've seen this with Together and Groq) may
    # include the reasoning in a <think></think> tag before the main response.
    # We expect this pattern to be repeated elsewhere, so include this hook to
    # automatically extract the reasoning content when the response is prefaced
    # with a <think> block. If this ends up being an overeach we can fall back
    # to each provider manually parsing out <think> using a helper function.
    # The implementation isn't important here, the critical thing to establish
    # is that EvalScope makes reasoning content available separately.
    @model_validator(mode='before')
    @classmethod
    def extract_reasoning(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # cleave apart <think> blocks
            content = data.get('content', None)
            if isinstance(content, str):
                content_text, content_reasoning = parse_content_with_reasoning(content)
                if content_reasoning:
                    data['content'] = [
                        content_reasoning,
                        ContentText(text=content_text),
                    ]
            # migrate messages that has explicit 'reasoning' field
            # (which was our original representation of reasoning)
            reasoning = data.get('reasoning', None)
            if isinstance(reasoning, str):
                # ensure that content is a list
                content = data.get('content', None)
                if content is None:
                    data['content'] = []
                elif isinstance(content, str):
                    data['content'] = [ContentText(text=content)]
                elif not isinstance(content, list):
                    data['content'] = []
                data['content'].insert(0, ContentReasoning(reasoning=reasoning))

                del data['reasoning']
        return data


class ChatMessageTool(ChatMessageBase):
    """Tool chat message."""

    role: Literal['tool'] = Field(default='tool')
    """Conversation role."""

    tool_call_id: Optional[str] = Field(default=None)
    """ID of tool call."""

    function: Optional[str] = Field(default=None)
    """Name of function called."""

    error: Optional[ToolCallError] = Field(default=None)
    """Error which occurred during tool call."""


ChatMessage = Union[ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool]
"""Message in a chat conversation"""


def dict_to_chat_message(data: Dict[str, Any]) -> ChatMessage:
    """Convert a dictionary to a ChatMessage."""

    if isinstance(data, ChatMessage):
        return data

    if 'role' not in data:
        raise ValueError('ChatMessage must have a "role" field')

    role = data['role']
    if role == 'system':
        return ChatMessageSystem.model_validate(data)
    elif role == 'user':
        return ChatMessageUser.model_validate(data)
    elif role == 'assistant':
        return ChatMessageAssistant.model_validate(data)
    elif role == 'tool':
        return ChatMessageTool.model_validate(data)
    else:
        raise ValueError(f'Unknown chat message role: {role}')


def messages_pretty_str(messages: List[ChatMessage]) -> str:
    """Pretty print a list of chat messages."""
    output = []
    for message in messages:
        role = message.role.capitalize()
        content = message.text
        if isinstance(message, ChatMessageTool):
            if message.error:
                content += f'\nError: {message.error.message}'
            if message.function:
                content += f'\nFunction: {message.function}'
        output.append(f'**{role}**: {content}')
    return '\n\n'.join(output)
