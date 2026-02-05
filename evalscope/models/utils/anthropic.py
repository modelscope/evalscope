import json
from anthropic import APIStatusError
from anthropic.types import (
    ContentBlock,
    ContentBlockParam,
    ImageBlockParam,
    Message,
    MessageParam,
    RedactedThinkingBlock,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceNoneParam,
    ToolChoiceParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from copy import copy
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentImage,
    ContentReasoning,
    ContentText,
)
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelOutput, ModelUsage, StopReason
from evalscope.api.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo, parse_tool_call
from evalscope.utils.url_utils import data_uri_mime_type, data_uri_to_base64, file_as_data_uri, is_http_url

BASE_64_DATA_REMOVED = '<base64-data-removed>'
NO_CONTENT = '[No content]'


class AnthropicResponseError(Exception):
    """Custom exception for Anthropic API response errors."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f'{self.code}: {self.message}'


def anthropic_tool_param(tool: ToolInfo) -> ToolParam:
    """Convert ToolInfo to Anthropic ToolParam."""
    return ToolParam(
        name=tool.name,
        description=tool.description,
        input_schema=tool.parameters.model_dump(exclude_none=True),
    )


def anthropic_chat_tools(tools: List[ToolInfo]) -> List[ToolParam]:
    """Convert list of ToolInfo to list of Anthropic ToolParam."""
    return [anthropic_tool_param(tool) for tool in tools]


def anthropic_chat_tool_choice(tool_choice: ToolChoice) -> ToolChoiceParam:
    """Convert ToolChoice to Anthropic ToolChoiceParam."""
    if tool_choice == 'none':
        return ToolChoiceNoneParam(type='none')
    elif tool_choice == 'any':
        return ToolChoiceAnyParam(type='any')
    elif isinstance(tool_choice, ToolFunction):
        return ToolChoiceToolParam(type='tool', name=tool_choice.name)
    else:  # 'auto'
        return ToolChoiceAutoParam(type='auto')


def anthropic_image_block_param(image: str) -> ImageBlockParam:
    """Convert image path/URL to Anthropic ImageBlockParam."""
    # Resolve to data URI if needed
    if not is_http_url(image) and not image.startswith('data:'):
        image = file_as_data_uri(image)

    # Get media type and base64 content
    media_type = data_uri_mime_type(image) or 'image/png'
    image_data = data_uri_to_base64(image)

    return ImageBlockParam(
        type='image',
        source=dict(type='base64', media_type=cast(Any, media_type), data=image_data),
    )


def anthropic_content_block_param(content: Content) -> List[ContentBlockParam]:
    """Convert Content to list of Anthropic ContentBlockParam."""
    if content.type == 'text':
        return [TextBlockParam(type='text', text=content.text or NO_CONTENT)]
    elif content.type == 'image':
        return [anthropic_image_block_param(content.image)]
    elif content.type == 'reasoning':
        # For reasoning content, we convert to text with think tags
        reasoning_text = f'<think>\n{content.reasoning}\n</think>'
        return [TextBlockParam(type='text', text=reasoning_text)]
    else:
        # For other content types (audio, video), convert to text representation
        return [TextBlockParam(type='text', text=f'[{content.type} content not supported]')]


def anthropic_message_param(message: ChatMessage) -> MessageParam:
    """Convert ChatMessage to Anthropic MessageParam.

    Note: Anthropic doesn't have a system role in messages - system messages
    should be passed separately to the API.
    """
    # Handle empty content
    content: Union[str, List[ContentBlockParam]]

    if message.role == 'system':
        # System messages are handled separately, but we convert them here for completeness
        raise ValueError('System messages should be handled separately in Anthropic API')

    elif message.role == 'tool':
        # Tool messages become user messages with tool_result blocks
        tool_message = cast(ChatMessageTool, message)
        if tool_message.error is not None:
            result_content: Union[str, List[TextBlockParam | ImageBlockParam]] = tool_message.error.message or 'error'
        elif isinstance(tool_message.content, str):
            result_content = [TextBlockParam(type='text', text=tool_message.content or NO_CONTENT)]
        else:
            result_content = []
            for c in tool_message.content:
                result_content.extend(anthropic_content_block_param(c))

        return MessageParam(
            role='user',
            content=[
                ToolResultBlockParam(
                    tool_use_id=str(tool_message.tool_call_id),
                    type='tool_result',
                    content=cast(List[TextBlockParam | ImageBlockParam], result_content),
                    is_error=tool_message.error is not None,
                )
            ],
        )

    elif message.role == 'assistant':
        assistant_message = cast(ChatMessageAssistant, message)
        block_params: List[ContentBlockParam] = []

        # Add content blocks
        if isinstance(assistant_message.content, str):
            if assistant_message.content:
                block_params.append(TextBlockParam(type='text', text=assistant_message.content))
        else:
            for c in assistant_message.content:
                block_params.extend(anthropic_content_block_param(c))

        # Add tool use blocks
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                block_params.append(
                    ToolUseBlockParam(
                        type='tool_use',
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_call.function.arguments,
                    )
                )

        return MessageParam(role='assistant', content=block_params or [TextBlockParam(type='text', text=NO_CONTENT)])

    else:  # user message
        user_message = cast(ChatMessageUser, message)
        if isinstance(user_message.content, str):
            content = user_message.content or NO_CONTENT
        else:
            content = []
            for c in user_message.content:
                content.extend(anthropic_content_block_param(c))
            if not content:
                content = [TextBlockParam(type='text', text=NO_CONTENT)]

        return MessageParam(role='user', content=content)


def anthropic_chat_messages(messages: List[ChatMessage], ) -> Tuple[Optional[str], List[MessageParam]]:
    """Convert list of ChatMessages to Anthropic format.

    Returns:
        Tuple of (system_message, message_params)
    """
    system_message: Optional[str] = None
    message_params: List[MessageParam] = []

    for message in messages:
        if message.role == 'system':
            # Anthropic requires system message as a separate parameter
            if system_message is None:
                system_message = message.text
            else:
                system_message = f'{system_message}\n\n{message.text}'
        else:
            message_params.append(anthropic_message_param(message))

    # Collapse consecutive user messages (required by Anthropic)
    message_params = _collapse_consecutive_messages(message_params, 'user')

    return system_message, message_params


def _collapse_consecutive_messages(messages: List[MessageParam], role: Literal['user',
                                                                               'assistant']) -> List[MessageParam]:
    """Collapse consecutive messages of the same role."""
    if not messages:
        return messages

    result: List[MessageParam] = []
    for message in messages:
        if message['role'] == role and result and result[-1]['role'] == role:
            # Combine with previous message
            result[-1] = _combine_messages(result[-1], message)
        else:
            result.append(message)

    return result


def _combine_messages(a: MessageParam, b: MessageParam) -> MessageParam:
    """Combine two messages of the same role."""
    role = a['role']
    a_content = a['content']
    b_content = b['content']

    if isinstance(a_content, str) and isinstance(b_content, str):
        return MessageParam(role=role, content=f'{a_content}\n{b_content}')
    elif isinstance(a_content, list) and isinstance(b_content, list):
        return MessageParam(role=role, content=a_content + b_content)
    elif isinstance(a_content, str) and isinstance(b_content, list):
        return MessageParam(role=role, content=[TextBlockParam(type='text', text=a_content)] + b_content)
    elif isinstance(a_content, list) and isinstance(b_content, str):
        return MessageParam(role=role, content=a_content + [TextBlockParam(type='text', text=b_content)])
    else:
        raise ValueError(f'Unexpected content types for messages: {a}, {b}')


def anthropic_completion_params(model: str, config: GenerateConfig) -> Dict[str, Any]:
    """Build Anthropic API completion parameters from GenerateConfig."""
    # Anthropic requires max_tokens to be set
    max_tokens = config.max_tokens if config.max_tokens is not None else 4096

    params: Dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
    )

    # Temperature (not compatible with extended thinking)
    if config.temperature is not None:
        params['temperature'] = config.temperature

    # Top P
    if config.top_p is not None:
        params['top_p'] = config.top_p

    # Top K
    if config.top_k is not None:
        params['top_k'] = config.top_k

    # Stop sequences
    if config.stop_seqs is not None:
        params['stop_sequences'] = config.stop_seqs

    # Timeout
    if config.timeout is not None:
        params['timeout'] = config.timeout

    # Extended thinking / reasoning
    if config.reasoning_tokens is not None:
        params['thinking'] = dict(type='enabled', budget_tokens=config.reasoning_tokens)

    # Extra body parameters
    if config.extra_body:
        for key, value in config.extra_body.items():
            if key not in params:
                params[key] = value

    return params


def chat_message_assistant_from_anthropic(model: str, message: Message, tools: List[ToolInfo]) -> ChatMessageAssistant:
    """Convert Anthropic Message to ChatMessageAssistant."""
    content, tool_calls = _content_and_tool_calls_from_blocks(message.content, tools)

    return ChatMessageAssistant(
        content=content,
        model=model,
        source='generate',
        tool_calls=tool_calls,
    )


def _content_and_tool_calls_from_blocks(
    content_blocks: Sequence[ContentBlock], tools: List[ToolInfo]
) -> Tuple[Union[str, List[Content]], Optional[List[ToolCall]]]:
    """Extract content and tool calls from Anthropic content blocks."""
    content: List[Content] = []
    tool_calls: Optional[List[ToolCall]] = None

    for block in content_blocks:
        if isinstance(block, TextBlock):
            if block.text:
                content.append(ContentText(text=block.text))
        elif isinstance(block, ToolUseBlock):
            tool_calls = tool_calls or []
            tool_calls.append(
                parse_tool_call(
                    block.id,
                    block.name,
                    json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    tools,
                )
            )
        elif isinstance(block, ThinkingBlock):
            content.append(ContentReasoning(reasoning=block.thinking, signature=block.signature))
        elif isinstance(block, RedactedThinkingBlock):
            content.append(ContentReasoning(reasoning=block.data, redacted=True))

    # If only one text content, return as string
    if len(content) == 1 and isinstance(content[0], ContentText):
        return content[0].text, tool_calls

    return content if content else '', tool_calls


def message_stop_reason(message: Message) -> StopReason:
    """Convert Anthropic stop reason to EvalScope StopReason."""
    match message.stop_reason:
        case 'end_turn' | 'stop_sequence':
            return 'stop'
        case 'tool_use':
            return 'tool_calls'
        case 'max_tokens':
            return 'model_length'
        case 'refusal':
            return 'content_filter'
        case _:
            return 'unknown'


def chat_choices_from_anthropic(message: Message, tools: List[ToolInfo]) -> List[ChatCompletionChoice]:
    """Convert Anthropic Message to list of ChatCompletionChoice."""
    assistant_message = chat_message_assistant_from_anthropic(message.model, message, tools)
    stop_reason = message_stop_reason(message)

    return [ChatCompletionChoice(
        message=assistant_message,
        stop_reason=stop_reason,
    )]


def model_output_from_anthropic(
    message: Message,
    choices: List[ChatCompletionChoice],
) -> ModelOutput:
    """Convert Anthropic Message to ModelOutput."""
    usage = message.usage.model_dump()
    input_tokens_cache_write = usage.get('cache_creation_input_tokens', None)
    input_tokens_cache_read = usage.get('cache_read_input_tokens', None)

    total_tokens = (
        message.usage.input_tokens + (input_tokens_cache_write or 0) +
        (input_tokens_cache_read or 0) + message.usage.output_tokens
    )

    return ModelOutput(
        model=message.model,
        choices=choices,
        usage=ModelUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            total_tokens=total_tokens,
            input_tokens_cache_write=input_tokens_cache_write,
            input_tokens_cache_read=input_tokens_cache_read,
        ),
    )


def anthropic_handle_bad_request(model_name: str, ex: APIStatusError) -> Union[ModelOutput, Exception]:
    """Handle Anthropic BadRequestError and convert to ModelOutput if possible."""
    error_message = str(ex.message).lower() if hasattr(ex, 'message') else str(ex).lower()
    content: Optional[str] = None
    stop_reason: Optional[StopReason] = None

    # Check for context length errors
    if any(
        msg in error_message for msg in [
            'prompt is too long',
            'input is too long',
            'input length and `max_tokens` exceed context limit',
        ]
    ):
        if isinstance(ex.body, dict) and 'error' in ex.body:
            error_dict = ex.body.get('error', {})
            if isinstance(error_dict, dict) and 'message' in error_dict:
                content = str(error_dict.get('message'))
            else:
                content = str(error_dict)
        else:
            content = error_message
        stop_reason = 'model_length'

    # Check for content filter errors
    elif 'content filtering' in error_message:
        content = 'Sorry, but I am unable to help with that request.'
        stop_reason = 'content_filter'

    if content and stop_reason:
        return ModelOutput.from_content(
            model=model_name,
            content=content,
            stop_reason=stop_reason,
        )
    else:
        return ex


def anthropic_media_filter(key: Optional[Any], value: Any) -> Any:
    """Filter out base64 encoded media from API call logs."""
    if key == 'source' and isinstance(value, dict) and value.get('type', None) == 'base64':
        value = copy(value)
        value.update(data=BASE_64_DATA_REMOVED)
    return value


def collect_stream_response(response_stream: Any) -> Message:
    """Collect streaming response chunks into a single Message.

    This function handles Anthropic's streaming response format.
    """
    from anthropic.types import (
        ContentBlockDeltaEvent,
        ContentBlockStartEvent,
        MessageDeltaEvent,
        MessageStartEvent,
        Usage,
    )

    message_id: str = ''
    model: str = ''
    role: str = 'assistant'
    content_blocks: List[ContentBlock] = []
    stop_reason: Optional[str] = None
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0

    current_block_index: int = -1
    current_text: str = ''
    current_tool_input: str = ''
    current_tool_id: str = ''
    current_tool_name: str = ''

    for event in response_stream:
        if isinstance(event, MessageStartEvent):
            message_id = event.message.id
            model = event.message.model
            role = event.message.role
            if event.message.usage:
                usage_input_tokens = event.message.usage.input_tokens

        elif isinstance(event, ContentBlockStartEvent):
            # Save previous block if exists
            if current_block_index >= 0:
                _append_content_block(
                    content_blocks, current_text, current_tool_id, current_tool_name, current_tool_input
                )

            current_block_index = event.index
            current_text = ''
            current_tool_input = ''
            current_tool_id = ''
            current_tool_name = ''

            if hasattr(event.content_block, 'text'):
                current_text = event.content_block.text or ''
            elif hasattr(event.content_block, 'id'):
                current_tool_id = event.content_block.id
                current_tool_name = getattr(event.content_block, 'name', '')

        elif isinstance(event, ContentBlockDeltaEvent):
            if hasattr(event.delta, 'text'):
                current_text += event.delta.text or ''
            elif hasattr(event.delta, 'partial_json'):
                current_tool_input += event.delta.partial_json or ''

        elif isinstance(event, MessageDeltaEvent):
            stop_reason = event.delta.stop_reason
            if event.usage:
                usage_output_tokens = event.usage.output_tokens

    # Append the last block
    if current_block_index >= 0:
        _append_content_block(content_blocks, current_text, current_tool_id, current_tool_name, current_tool_input)

    return Message(
        id=message_id,
        type='message',
        role=role,  # type: ignore
        model=model,
        content=content_blocks,
        stop_reason=stop_reason,  # type: ignore
        stop_sequence=None,
        usage=Usage(input_tokens=usage_input_tokens, output_tokens=usage_output_tokens),
    )


def _append_content_block(
    content_blocks: List[ContentBlock],
    text: str,
    tool_id: str,
    tool_name: str,
    tool_input: str,
) -> None:
    """Helper to append a content block to the list."""
    if tool_id:
        # Tool use block
        try:
            input_data = json.loads(tool_input) if tool_input else {}
        except json.JSONDecodeError:
            input_data = {}
        content_blocks.append(ToolUseBlock(type='tool_use', id=tool_id, name=tool_name, input=input_data))
    elif text:
        # Text block
        content_blocks.append(TextBlock(type='text', text=text))
