import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    Content,
    ContentReasoning,
    ContentText,
)
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelOutput, ModelUsage, StopReason
from evalscope.api.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo, parse_tool_call
from evalscope.utils.url_utils import file_as_data_uri, is_data_uri, is_http_url
from .openai import openai_assistant_content

try:
    from openai.types.responses import Response
except ImportError:
    Response = Any


def openai_response_input_part(content: Content) -> Dict[str, Any]:
    if content.type == 'text':
        return {'type': 'input_text', 'text': content.text}
    if content.type == 'image':
        image_url = content.image
        if not is_http_url(image_url) and not is_data_uri(image_url):
            image_url = file_as_data_uri(image_url)
        return {'type': 'input_image', 'image_url': image_url, 'detail': content.detail}
    if content.type == 'audio':
        raise RuntimeError('Audio content is not currently supported by OpenAI Responses API input messages.')
    if content.type == 'video':
        raise RuntimeError('Video content is not currently supported by OpenAI Responses API input messages.')
    raise RuntimeError(f'Content type {content.type} is not currently supported by OpenAI Responses API.')


def openai_response_message(message: ChatMessage) -> List[Dict[str, Any]]:
    if message.role == 'system':
        return [{'role': 'system', 'content': message.text}]
    if message.role == 'user':
        return [{
            'role': 'user',
            'content': (
                message.content if isinstance(message.content, str) else
                [openai_response_input_part(content) for content in message.content]
            ),
        }]
    if message.role == 'assistant':
        return openai_response_assistant_message(message)
    if message.role == 'tool':
        return [openai_response_tool_message(message)]
    raise ValueError(f'Unexpected message role {message.role}')


def openai_response_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for message in messages:
        items.extend(openai_response_message(message))
    return items


def openai_response_assistant_message(message: ChatMessageAssistant) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    content = openai_assistant_content(message)
    if content:
        items.append({'role': 'assistant', 'content': content})
    for call in message.tool_calls or []:
        items.append(openai_response_function_call(call))
    return items


def openai_response_tool_message(message: ChatMessageTool) -> Dict[str, Any]:
    return {
        'type': 'function_call_output',
        'call_id': str(message.tool_call_id),
        'output': f'Error: {message.error.message}' if message.error else message.text,
    }


def openai_response_function_call(tool_call: ToolCall) -> Dict[str, Any]:
    return {
        'type': 'function_call',
        'call_id': tool_call.id,
        'name': tool_call.function.name,
        'arguments': json.dumps(tool_call.function.arguments),
    }


def openai_response_tool_param(tool: ToolInfo) -> Dict[str, Any]:
    return {
        'type': 'function',
        'name': tool.name,
        'description': tool.description,
        'parameters': tool.parameters.model_dump(exclude_none=True),
        'strict': False,
    }


def openai_response_tools(tools: List[ToolInfo]) -> List[Dict[str, Any]]:
    return [openai_response_tool_param(tool) for tool in tools]


def openai_response_tool_choice(tool_choice: ToolChoice) -> Union[str, Dict[str, Any]]:
    if isinstance(tool_choice, ToolFunction):
        return {'type': 'function', 'name': tool_choice.name}
    if tool_choice == 'any':
        return 'required'
    return tool_choice


def openai_response_params(model: str, config: GenerateConfig, tools: bool) -> Dict[str, Any]:
    params: Dict[str, Any] = {'model': model}
    if config.stream is not None:
        params['stream'] = config.stream
    if config.timeout is not None:
        params['timeout'] = config.timeout
    if config.max_tokens is not None:
        params['max_output_tokens'] = config.max_tokens
    if config.temperature is not None:
        params['temperature'] = config.temperature
    if config.top_p is not None:
        params['top_p'] = config.top_p
    if config.top_logprobs is not None:
        params['top_logprobs'] = config.top_logprobs
    if tools and config.parallel_tool_calls is not None:
        params['parallel_tool_calls'] = config.parallel_tool_calls
    if config.reasoning_effort is not None or config.reasoning_summary is not None:
        reasoning: Dict[str, Any] = {}
        if config.reasoning_effort is not None:
            reasoning['effort'] = config.reasoning_effort
        if config.reasoning_summary is not None:
            reasoning['summary'] = config.reasoning_summary
        params['reasoning'] = reasoning
    if config.response_schema is not None:
        text_format = {
            'type': 'json_schema',
            'name': config.response_schema.name,
            'schema': config.response_schema.json_schema.model_dump(exclude_none=True),
        }
        if config.response_schema.description is not None:
            text_format['description'] = config.response_schema.description
        if config.response_schema.strict is not None:
            text_format['strict'] = config.response_schema.strict
        params['text'] = {'format': text_format}
    if config.extra_body:
        params['extra_body'] = config.extra_body
    if config.extra_query:
        params['extra_query'] = config.extra_query
    if config.extra_headers:
        params['extra_headers'] = config.extra_headers
    return params


def collect_response_stream(
    response_stream: Iterable[Any],
    request_start: Optional[float] = None,
) -> Tuple[Response, Optional[float]]:
    response: Optional[Response] = None
    t_start = request_start if request_start is not None else time.monotonic()
    ttft: Optional[float] = None

    for event in response_stream:
        event_type = getattr(event, 'type', None)
        delta = getattr(event, 'delta', None)
        if ttft is None and event_type in _FIRST_TOKEN_EVENT_TYPES and delta:
            ttft = time.monotonic() - t_start
        if event_type in ('response.completed', 'response.incomplete', 'response.failed'):
            response = event.response

    if response is None:
        raise ValueError('OpenAI Responses stream ended without a terminal response event.')
    return response, ttft


async def async_collect_response_stream(
    response_stream: Any,
    request_start: Optional[float] = None,
) -> Tuple[Response, Optional[float]]:
    response: Optional[Response] = None
    t_start = request_start if request_start is not None else time.monotonic()
    ttft: Optional[float] = None

    async for event in response_stream:
        event_type = getattr(event, 'type', None)
        delta = getattr(event, 'delta', None)
        if ttft is None and event_type in _FIRST_TOKEN_EVENT_TYPES and delta:
            ttft = time.monotonic() - t_start
        if event_type in ('response.completed', 'response.incomplete', 'response.failed'):
            response = event.response

    if response is None:
        raise ValueError('OpenAI Responses stream ended without a terminal response event.')
    return response, ttft


def model_output_from_openai_response(
    response: Response,
    choices: List[ChatCompletionChoice],
) -> ModelOutput:
    return ModelOutput(
        id=response.id,
        model=response.model,
        choices=choices,
        usage=model_usage_from_openai_response(response),
    )


def model_usage_from_openai_response(response: Response) -> Optional[ModelUsage]:
    usage = response.usage
    if usage is None:
        return None
    return ModelUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        input_tokens_cache_read=(
            usage.input_tokens_details.cached_tokens if usage.input_tokens_details is not None else None
        ),
        reasoning_tokens=(
            usage.output_tokens_details.reasoning_tokens if usage.output_tokens_details is not None else None
        ),
        total_tokens=usage.total_tokens,
    )


def chat_choices_from_openai_response(response: Response, tools: List[ToolInfo]) -> List[ChatCompletionChoice]:
    text = response_output_text(response)
    reasoning = response_reasoning_text(response)
    tool_calls = response_tool_calls(response, tools)
    content: Union[str, List[Content]]
    if reasoning:
        content = [ContentReasoning(reasoning=reasoning), ContentText(text=text)]
    else:
        content = text
    return [
        ChatCompletionChoice(
            message=ChatMessageAssistant(
                content=content,
                model=response.model,
                source='generate',
                tool_calls=tool_calls or None,
            ),
            stop_reason=response_stop_reason(response, has_tool_calls=bool(tool_calls)),
        )
    ]


def response_output_text(response: Response) -> str:
    output_text = getattr(response, 'output_text', None)
    if isinstance(output_text, str) and output_text:
        return output_text

    parts: List[str] = []
    for item in response.output or []:
        if getattr(item, 'type', None) != 'message':
            continue
        for content in getattr(item, 'content', []) or []:
            content_type = getattr(content, 'type', None)
            if content_type == 'output_text':
                parts.append(getattr(content, 'text', '') or '')
            elif content_type == 'refusal':
                parts.append(getattr(content, 'refusal', '') or '')
    return ''.join(parts)


def response_reasoning_text(response: Response) -> str:
    parts: List[str] = []
    for item in response.output or []:
        if getattr(item, 'type', None) != 'reasoning':
            continue
        for summary in getattr(item, 'summary', []) or []:
            text = getattr(summary, 'text', None)
            if text:
                parts.append(text)
        for content in getattr(item, 'content', []) or []:
            text = getattr(content, 'text', None)
            if text:
                parts.append(text)
    return ''.join(parts)


def response_tool_calls(response: Response, tools: List[ToolInfo]) -> List[ToolCall]:
    calls: List[ToolCall] = []
    for item in response.output or []:
        if getattr(item, 'type', None) != 'function_call':
            continue
        calls.append(
            parse_tool_call(
                getattr(item, 'call_id', '') or getattr(item, 'id', ''),
                getattr(item, 'name', ''),
                getattr(item, 'arguments', '') or '{}',
                tools,
            )
        )
    return calls


def response_stop_reason(response: Response, has_tool_calls: bool = False) -> StopReason:
    if has_tool_calls:
        return 'tool_calls'
    if response.status == 'incomplete' and response.incomplete_details is not None:
        reason = response.incomplete_details.reason
        if reason == 'max_output_tokens':
            return 'max_tokens'
        if reason == 'content_filter':
            return 'content_filter'
    if response.status == 'failed':
        return 'unknown'
    return 'stop'


def response_usage_from_dict(payload: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    usage = payload.get('usage')
    if not usage:
        return None
    if 'input_tokens' in usage or 'output_tokens' in usage:
        return usage.get('input_tokens', 0), usage.get('output_tokens', 0)
    if 'prompt_tokens' in usage or 'completion_tokens' in usage:
        return usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    return None


def response_text_from_dict(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get('output_text'), str):
        return payload['output_text']
    parts: List[str] = []
    for item in payload.get('output', []) or []:
        if item.get('type') != 'message':
            continue
        for content in item.get('content', []) or []:
            content_type = content.get('type')
            if content_type == 'output_text':
                parts.append(content.get('text', '') or '')
            elif content_type == 'refusal':
                parts.append(content.get('refusal', '') or '')
    return ''.join(parts)


def normalize_responses_input(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_responses_message(message) if isinstance(message, dict) else message for message in value]
    return value


def normalize_responses_message(message: Dict[str, Any]) -> Dict[str, Any]:
    content = message.get('content')
    if not isinstance(content, list):
        return message
    normalized = dict(message)
    normalized['content'] = [normalize_responses_content_part(part) for part in content]
    return normalized


def normalize_responses_content_part(part: Dict[str, Any]) -> Dict[str, Any]:
    part_type = part.get('type')
    if part_type == 'text':
        return {'type': 'input_text', 'text': part.get('text', '')}
    if part_type == 'image_url':
        image_url = part.get('image_url', {})
        if isinstance(image_url, str):
            return {'type': 'input_image', 'image_url': image_url, 'detail': 'auto'}
        return {
            'type': 'input_image',
            'image_url': image_url.get('url'),
            'detail': image_url.get('detail', 'auto'),
        }
    return part


_FIRST_TOKEN_EVENT_TYPES = {
    'response.output_text.delta',
    'response.refusal.delta',
    'response.reasoning_text.delta',
    'response.reasoning_summary_text.delta',
    'response.function_call_arguments.delta',
}
