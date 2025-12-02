import base64
import json
import re
from collections import defaultdict
from copy import copy
from openai import APIStatusError, OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import JsonValue
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentAudio,
    ContentImage,
    ContentReasoning,
    ContentText,
    parse_content_with_reasoning,
)
from evalscope.api.model import (
    ChatCompletionChoice,
    GenerateConfig,
    Logprobs,
    ModelOutput,
    ModelUsage,
    StopReason,
    as_stop_reason,
)
from evalscope.api.tool import ToolCall, ToolChoice, ToolFunction, ToolInfo, parse_tool_call
from evalscope.utils.url_utils import file_as_data_uri, is_http_url

BASE_64_DATA_REMOVED = '<base64-data-removed>'


class OpenAIResponseError(OpenAIError):

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f'{self.code}: {self.message}'


def openai_chat_tool_call(tool_call: ToolCall) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        type='function',
        id=tool_call.id,
        function=Function(name=tool_call.function.name, arguments=json.dumps(tool_call.function.arguments)),
    )


def openai_chat_tool_call_param(tool_call: ToolCall) -> ChatCompletionMessageToolCallParam:
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        function=dict(name=tool_call.function.name, arguments=json.dumps(tool_call.function.arguments)),
        type='function',
    )


def openai_chat_completion_part(content: Content) -> ChatCompletionContentPartParam:
    if content.type == 'text':
        return ChatCompletionContentPartTextParam(type='text', text=content.text)
    elif content.type == 'image':
        # API takes URL or base64 encoded file. If it's a remote file or
        # data URL leave it alone, otherwise encode it
        image_url = content.image
        detail = content.detail

        if not is_http_url(image_url):
            image_url = file_as_data_uri(image_url)

        return ChatCompletionContentPartImageParam(
            type='image_url',
            image_url=dict(url=image_url, detail=detail),
        )
    elif content.type == 'audio':
        audio_data_uri = file_as_data_uri(content.audio)

        return ChatCompletionContentPartInputAudioParam(
            type='input_audio', input_audio=dict(data=audio_data_uri, format=content.format)
        )

    else:
        raise RuntimeError('Video content is not currently supported by Open AI chat models.')


def openai_chat_message(
    message: ChatMessage, system_role: Literal['user', 'system', 'developer'] = 'system'
) -> ChatCompletionMessageParam:
    if message.role == 'system':
        if system_role == 'user':
            return ChatCompletionUserMessageParam(role='user', content=message.text)
        elif system_role == 'system':
            return ChatCompletionSystemMessageParam(role=message.role, content=message.text)
        elif system_role == 'developer':
            return ChatCompletionDeveloperMessageParam(role='developer', content=message.text)
    elif message.role == 'user':
        return ChatCompletionUserMessageParam(
            role=message.role,
            content=(
                message.content if isinstance(message.content, str) else
                [openai_chat_completion_part(content) for content in message.content]
            ),
        )
    elif message.role == 'assistant':
        if message.tool_calls:
            return ChatCompletionAssistantMessageParam(
                role=message.role,
                content=openai_assistant_content(message),
                tool_calls=[openai_chat_tool_call_param(call) for call in message.tool_calls],
            )
        else:
            return ChatCompletionAssistantMessageParam(role=message.role, content=openai_assistant_content(message))
    elif message.role == 'tool':
        return ChatCompletionToolMessageParam(
            role=message.role,
            content=(f'Error: {message.error.message}' if message.error else message.text),
            tool_call_id=str(message.tool_call_id),
        )
    else:
        raise ValueError(f'Unexpected message role {message.role}')


def openai_chat_messages(
    messages: List[ChatMessage],
    system_role: Literal['user', 'system', 'developer'] = 'system',
) -> List[ChatCompletionMessageParam]:
    return [openai_chat_message(message, system_role) for message in messages]


def openai_completion_params(model: str, config: GenerateConfig, tools: bool) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(model=model)
    # handle stream option
    if config.stream is not None:
        params['stream'] = config.stream
        if config.stream:
            params['stream_options'] = {'include_usage': True}
    if config.timeout is not None:
        params['timeout'] = config.timeout
    if config.max_tokens is not None:
        params['max_tokens'] = config.max_tokens
    if config.frequency_penalty is not None:
        params['frequency_penalty'] = config.frequency_penalty
    if config.stop_seqs is not None:
        params['stop'] = config.stop_seqs
    if config.presence_penalty is not None:
        params['presence_penalty'] = config.presence_penalty
    if config.repetition_penalty is not None:
        params['repetition_penalty'] = config.repetition_penalty
    if config.logit_bias is not None:
        params['logit_bias'] = config.logit_bias
    if config.seed is not None:
        params['seed'] = config.seed
    if config.temperature is not None:
        params['temperature'] = config.temperature
    if config.top_p is not None:
        params['top_p'] = config.top_p
    if config.top_k is not None:
        params['top_k'] = config.top_k
    if config.n is not None:
        params['n'] = config.n
    if config.logprobs is not None:
        params['logprobs'] = config.logprobs
    if config.top_logprobs is not None:
        params['top_logprobs'] = config.top_logprobs
    if tools and config.parallel_tool_calls is not None:
        params['parallel_tool_calls'] = config.parallel_tool_calls
    if config.reasoning_effort is not None:
        params['reasoning_effort'] = config.reasoning_effort
    if config.response_schema is not None:
        params['response_format'] = dict(
            type='json_schema',
            json_schema=dict(
                name=config.response_schema.name,
                schema=config.response_schema.json_schema.model_dump(exclude_none=True),
                description=config.response_schema.description,
                strict=config.response_schema.strict,
            ),
        )
    if config.extra_body:
        params['extra_body'] = config.extra_body
    if config.extra_query:
        params['extra_query'] = config.extra_query
    if config.extra_headers:
        params['extra_headers'] = config.extra_headers

    return params


def openai_assistant_content(message: ChatMessageAssistant, include_reasoning=True) -> str:
    # In agent bridge scenarios, we could encounter concepts such as reasoning and
    # .internal use in the ChatMessageAssistant that are not supported by the OpenAI
    # choices API. This code smuggles that data into the plain text so that it
    # survives multi-turn round trips.

    if isinstance(message.content, str):
        content = message.content
    else:
        content = ''
        for c in message.content:
            if c.type == 'reasoning' and include_reasoning:
                attribs = ''
                if c.signature is not None:
                    attribs = f'{attribs} signature="{c.signature}"'
                if c.redacted:
                    attribs = f'{attribs} redacted="true"'
                content = f'{content}\n<think{attribs}>\n{c.reasoning}\n</think>\n'
            elif c.type == 'text':
                content = f'{content}\n{c.text}'

    if message.internal:
        content = f"""{content}\n<internal>{
            base64.b64encode(json.dumps(message.internal).encode("utf-8")).decode(
                "utf-8"
            )
        }</internal>\n"""
    return content


def openai_chat_choices(choices: List[ChatCompletionChoice], include_reasoning: bool = True) -> List[Choice]:
    oai_choices: List[Choice] = []

    for index, choice in enumerate(choices):
        # Handle content
        content = openai_assistant_content(choice.message, include_reasoning=include_reasoning)

        # Handle tool calls
        if choice.message.tool_calls:
            tool_calls = [openai_chat_tool_call(tc) for tc in choice.message.tool_calls]
        else:
            tool_calls = None
        message = ChatCompletionMessage(role='assistant', content=content, tool_calls=tool_calls)
        oai_choices.append(
            Choice(
                finish_reason=openai_finish_reason(choice.stop_reason),
                index=index,
                message=message,
                logprobs=ChoiceLogprobs(**choice.logprobs.model_dump()) if choice.logprobs is not None else None,
            )
        )

    return oai_choices


def openai_completion_usage(usage: ModelUsage) -> CompletionUsage:
    return CompletionUsage(
        completion_tokens=usage.output_tokens,
        prompt_tokens=usage.input_tokens,
        total_tokens=usage.total_tokens,
    )


def openai_finish_reason(
    stop_reason: StopReason
) -> Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']:
    if stop_reason in ('stop', 'tool_calls', 'content_filter'):
        return stop_reason
    elif stop_reason == 'model_length':
        return 'length'
    else:
        return 'stop'


def openai_chat_tool_param(tool: ToolInfo) -> ChatCompletionToolParam:
    function = FunctionDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters.model_dump(exclude_none=True),
    )
    return ChatCompletionToolParam(type='function', function=function)


def openai_chat_tools(tools: List[ToolInfo]) -> List[ChatCompletionToolParam]:
    return [openai_chat_tool_param(tool) for tool in tools]


def openai_chat_tool_choice(tool_choice: ToolChoice, ) -> ChatCompletionToolChoiceOptionParam:
    if isinstance(tool_choice, ToolFunction):
        return ChatCompletionNamedToolChoiceParam(type='function', function=dict(name=tool_choice.name))
    # openai supports 'any' via the 'required' keyword
    elif tool_choice == 'any':
        return 'required'
    else:
        return tool_choice


def chat_tool_calls_from_openai(message: ChatCompletionMessage, tools: List[ToolInfo]) -> Optional[List[ToolCall]]:
    if message.tool_calls:
        return [
            parse_tool_call(call.id, call.function.name, call.function.arguments, tools) for call in message.tool_calls
        ]
    else:
        return None


def chat_messages_from_openai(
    model: str,
    messages: List[ChatCompletionMessageParam],
) -> List[ChatMessage]:
    # track tool names by id
    tool_names: Dict[str, str] = {}

    chat_messages: List[ChatMessage] = []

    for message in messages:
        content: Union[str, List[Content]] = []
        if message['role'] == 'system' or message['role'] == 'developer':
            sys_content = message['content']
            if isinstance(sys_content, str):
                chat_messages.append(ChatMessageSystem(content=sys_content))
            else:
                content = []
                for sc in sys_content:
                    content.extend(content_from_openai(sc))
                chat_messages.append(ChatMessageSystem(content=content))
        elif message['role'] == 'user':
            user_content = message['content']
            if isinstance(user_content, str):
                chat_messages.append(ChatMessageUser(content=user_content))
            else:
                content = []
                for uc in user_content:
                    content.extend(content_from_openai(uc))
                chat_messages.append(ChatMessageUser(content=content))
        elif message['role'] == 'assistant':
            # resolve content
            refusal: Optional[Literal[True]] = None
            internal: Optional[JsonValue] = None
            asst_content = message.get('content', None)
            if isinstance(asst_content, str):
                # Even though the choices API doesn't take advantage of .internal,
                # we could be transforming from OpenAI choices to Inspect for agent
                # bridge scenarios where a different model (that does use .internal)
                # is the actual model being used.
                asst_content, internal = _parse_content_with_internal(asst_content)
                asst_content, smuggled_reasoning = parse_content_with_reasoning(asst_content)
                if smuggled_reasoning:
                    content = [
                        smuggled_reasoning,
                        ContentText(text=asst_content),
                    ]
                else:
                    content = asst_content
            elif asst_content is None:
                content = message.get('refusal', None) or ''
                if content:
                    refusal = True
            else:
                content = []
                for ac in asst_content:
                    content.extend(content_from_openai(ac, parse_reasoning=True))

            # resolve reasoning (OpenAI doesn't suport this however OpenAI-compatible
            # interfaces e.g. DeepSeek do include this field so we pluck it out)
            reasoning = message.get('reasoning_content', None) or message.get('reasoning', None)
            if reasoning is not None:
                # normalize content to an array
                if isinstance(content, str):
                    content = [ContentText(text=content, refusal=refusal)]

                # insert reasoning
                content.insert(0, ContentReasoning(reasoning=str(reasoning)))

            # return message
            if 'tool_calls' in message:
                tool_calls: List[ToolCall] = []
                for call in message['tool_calls']:
                    tool_calls.append(tool_call_from_openai(call))
                    tool_names[call['id']] = call['function']['name']

            else:
                tool_calls = []

            chat_messages.append(
                ChatMessageAssistant(
                    content=content,
                    tool_calls=tool_calls or None,
                    model=model,
                    source='generate',
                    internal=internal,
                )
            )
        elif message['role'] == 'tool':
            tool_content = message.get('content', None) or ''
            if isinstance(tool_content, str):
                # If tool_content is a simple str, it could be the result of some
                # sub-agent tool call that has <think> or <internal> smuggled inside
                # of it to support agent bridge scenarios. We have to strip that
                # data. To be clear, if it's <think>, we'll strip the <think> tag,
                # but the reasoning summary itself will remain in the content.
                content, _ = _parse_content_with_internal(tool_content)
                content, _ = parse_content_with_reasoning(content)
            else:
                content = []
                for tc in tool_content:
                    content.extend(content_from_openai(tc))
            chat_messages.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=message['tool_call_id'],
                    function=tool_names.get(message['tool_call_id'], ''),
                )
            )
        else:
            raise ValueError(f'Unexpected message param type: {type(message)}')

    return chat_messages


def tool_call_from_openai(tool_call: ChatCompletionMessageToolCallParam) -> ToolCall:
    return parse_tool_call(
        tool_call['id'],
        tool_call['function']['name'],
        tool_call['function']['arguments'],
    )


def content_from_openai(
    content: Union[ChatCompletionContentPartParam, ChatCompletionContentPartRefusalParam],
    parse_reasoning: bool = False,
) -> List[Content]:
    # Some providers omit the type tag and use "object-with-a-single-field" encoding
    if 'type' not in content and len(content) == 1:
        content['type'] = list(content.keys())[0]  # type: ignore[arg-type]
    if content['type'] == 'text':
        text = content['text']
        if parse_reasoning:
            content_text, content_reasoning = parse_content_with_reasoning(text)
            if content_reasoning:
                return [
                    content_reasoning,
                    ContentText(text=content_text),
                ]
            else:
                return [ContentText(text=text)]
        else:
            return [ContentText(text=text)]
    elif content['type'] == 'reasoning':  # type: ignore[comparison-overlap]
        return [ContentReasoning(reasoning=content['reasoning'])]
    elif content['type'] == 'image_url':
        return [ContentImage(image=content['image_url']['url'], detail=content['image_url'].get('detail', 'auto'))]
    elif content['type'] == 'input_audio':
        return [ContentAudio(
            audio=content['input_audio']['data'],
            format=content['input_audio']['format'],
        )]
    elif content['type'] == 'refusal':
        return [ContentText(text=content['refusal'], refusal=True)]
    else:
        content_type = content['type']
        raise ValueError(f"Unexpected content type '{content_type}' in message.")


def chat_message_assistant_from_openai(
    model: str, message: ChatCompletionMessage, tools: List[ToolInfo]
) -> ChatMessageAssistant:
    refusal = getattr(message, 'refusal', None)
    reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)

    msg_content = refusal or message.content or ''
    if reasoning is not None:
        content: Union[str, List[Content]] = [
            ContentReasoning(reasoning=str(reasoning)),
            ContentText(text=msg_content, refusal=True if refusal else None),
        ]
    elif refusal is not None:
        content = [ContentText(text=msg_content, refusal=True)]
    else:
        content = msg_content

    return ChatMessageAssistant(
        content=content,
        model=model,
        source='generate',
        tool_calls=chat_tool_calls_from_openai(message, tools),
    )


def model_output_from_openai(
    completion: ChatCompletion,
    choices: list[ChatCompletionChoice],
) -> ModelOutput:
    return ModelOutput(
        model=completion.model,
        choices=choices,
        usage=(
            ModelUsage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                input_tokens_cache_read=(
                    completion.usage.prompt_tokens_details.cached_tokens if completion.usage.prompt_tokens_details
                    is not None else None  # openai only have cache read stats/pricing.
                ),
                reasoning_tokens=(
                    completion.usage.completion_tokens_details.reasoning_tokens
                    if completion.usage.completion_tokens_details is not None else None
                ),
                total_tokens=completion.usage.total_tokens,
            ) if completion.usage else None
        ),
    )


def chat_choices_from_openai(response: ChatCompletion, tools: List[ToolInfo]) -> List[ChatCompletionChoice]:
    choices = list(response.choices)
    choices.sort(key=lambda c: c.index)
    return [
        ChatCompletionChoice(
            message=chat_message_assistant_from_openai(response.model, choice.message, tools),
            stop_reason=as_stop_reason(choice.finish_reason),
            logprobs=(
                Logprobs(**choice.logprobs.model_dump())
                if choice.logprobs and choice.logprobs.content is not None else None
            ),
        ) for choice in choices
    ]


def openai_handle_bad_request(model_name: str, e: APIStatusError) -> Union[ModelOutput, Exception]:
    # extract message
    if isinstance(e.body, dict) and 'message' in e.body.keys():
        content = str(e.body.get('message'))
    else:
        content = e.message

    # narrow stop_reason
    stop_reason: Optional[StopReason] = None
    if e.code == 'context_length_exceeded':
        stop_reason = 'model_length'
    elif (
        e.code == 'invalid_prompt'  # seems to happen for o1/o3
        or e.code == 'content_policy_violation'  # seems to happen for vision
        or e.code == 'content_filter'  # seems to happen on azure
    ):
        stop_reason = 'content_filter'

    if stop_reason:
        return ModelOutput.from_content(model=model_name, content=content, stop_reason=stop_reason)
    else:
        raise e


def openai_media_filter(key: Optional[JsonValue], value: JsonValue) -> JsonValue:
    # remove images from raw api call
    if key == 'output' and isinstance(value, dict) and 'image_url' in value:
        value = copy(value)
        value.update(image_url=BASE_64_DATA_REMOVED)
    if key == 'image_url' and isinstance(value, dict) and 'url' in value:
        url = str(value.get('url'))
        if url.startswith('data:'):
            value = copy(value)
            value.update(url=BASE_64_DATA_REMOVED)
    elif key == 'input_audio' and isinstance(value, dict) and 'data' in value:
        value = copy(value)
        value.update(data=BASE_64_DATA_REMOVED)
    return value


def _parse_content_with_internal(content: str, ) -> Tuple[str, Optional[JsonValue]]:
    """
    Extracts and removes a smuggled <internal>...</internal> tag from the content string, if present.

    Note:
        This OpenAI model does not natively use `.internal`. However, in bridge
        scenarios—where output from a model that does use `.internal` is routed
        through this code—such a tag may be present and should be handled.

    Args:
        content: The input string, possibly containing an <internal> tag with
        base64-encoded JSON.

    Returns:
        tuple[str, JsonValue | None]:
            - The content string with the <internal>...</internal> tag removed (if present), otherwise the original string.
            - The decoded and parsed internal value (if present), otherwise None.

    Raises:
        json.JSONDecodeError: If the content of the <internal> tag is not valid JSON after decoding.
        UnicodeDecodeError: If the content of the <internal> tag is not valid UTF-8 after base64 decoding.
    """  # noqa: E501
    internal_pattern = r'<internal>(.*?)</internal>'
    internal_match = re.search(r'<internal>(.*?)</internal>', content, re.DOTALL)

    return ((
        re.sub(internal_pattern, '', content, flags=re.DOTALL).strip(),
        json.loads(base64.b64decode(internal_match.group(1)).decode('utf-8')),
    ) if internal_match else (content, None))


def collect_stream_response(response_stream: List[ChatCompletionChunk]) -> ChatCompletion:
    collected_chunks: List[ChatCompletionChunk] = []
    collected_messages = defaultdict(list)
    collected_reasoning = defaultdict(list)
    collected_tool_calls = defaultdict(dict)

    for chunk in response_stream:
        collected_chunks.append(chunk)
        for choice in chunk.choices:
            # Handle reasoning content
            if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content is not None:
                collected_reasoning[choice.index].append(choice.delta.reasoning_content)

            # Handle regular content
            if choice.delta.content is not None:
                collected_messages[choice.index].append(choice.delta.content)

            # Handle tool calls
            if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    tool_id = tool_call.index

                    # Initialize tool call if not present
                    if tool_id not in collected_tool_calls[choice.index]:
                        collected_tool_calls[choice.index][tool_id] = {
                            'id': tool_call.id if hasattr(tool_call, 'id') and tool_call.id else None,
                            'type': tool_call.type if hasattr(tool_call, 'type') and tool_call.type else None,
                            'function': {
                                'name': '',
                                'arguments': ''
                            }
                        }

                    # Update tool call with new chunks
                    if hasattr(tool_call, 'function'):
                        if hasattr(tool_call.function, 'name') and tool_call.function.name:
                            collected_tool_calls[choice.index][tool_id]['function']['name'] = tool_call.function.name

                        if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                            collected_tool_calls[choice.index
                                                 ][tool_id]['function']['arguments'] += tool_call.function.arguments

                    # Update ID if it was received later
                    if hasattr(tool_call, 'id') and tool_call.id:
                        collected_tool_calls[choice.index][tool_id]['id'] = tool_call.id

    # Get all unique choice indices from all collections
    all_indices = set(collected_messages.keys()) | set(collected_reasoning.keys()) | set(collected_tool_calls.keys())

    choices = []
    for index in all_indices:
        full_reply_content = ''.join(collected_messages.get(index, []))
        reasoning = ''.join(collected_reasoning.get(index, []))

        # Process tool_calls for this choice if any exists
        tool_calls_list = None
        if index in collected_tool_calls and collected_tool_calls[index]:
            tool_calls_list = list(collected_tool_calls[index].values())
            # Filter out any tool calls with None id (incomplete tool calls)
            tool_calls_list = [tc for tc in tool_calls_list if tc['id'] is not None]

        # use the finish_reason from the last chunk that generated this choice
        finish_reason = None
        for chunk in reversed(collected_chunks):
            if chunk.choices and chunk.choices[0].index == index:
                finish_reason = chunk.choices[0].finish_reason
                break

        message_kwargs = {'role': 'assistant', 'content': full_reply_content}

        if reasoning:
            message_kwargs['reasoning_content'] = reasoning

        if tool_calls_list:
            message_kwargs['tool_calls'] = tool_calls_list

        choice = Choice(
            finish_reason=finish_reason or 'stop', index=index, message=ChatCompletionMessage(**message_kwargs)
        )
        choices.append(choice)

    # build the final completion object
    return ChatCompletion(
        id=collected_chunks[0].id,
        choices=choices,
        created=collected_chunks[0].created,
        model=collected_chunks[0].model,
        object='chat.completion',
        usage=collected_chunks[-1].usage  # use the usage from the last chunk
    )
