"""LiteLLM model API provider for EvalScope.

Routes to 100+ LLM providers via litellm.completion().
Provider API keys are read from environment variables automatically
(OPENAI_API_KEY, ANTHROPIC_API_KEY, AWS_ACCESS_KEY_ID, GEMINI_API_KEY, etc.).

Model names use LiteLLM format: "provider/model-name".
See https://docs.litellm.ai/docs/providers for the full list.
"""

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, Dict, List, Optional, Tuple, Type

import litellm
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from evalscope.api.messages import ChatMessage
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.function_utils import async_retry_call, retry_call

from .utils.openai import (
    async_collect_stream_response,
    chat_choices_from_openai,
    collect_stream_response,
    model_output_from_openai,
    openai_chat_messages,
    openai_chat_tool_choice,
    openai_chat_tools,
    openai_completion_params,
    openai_handle_bad_request,
)

logger = get_logger()


# LiteLLM maps provider client errors (context length exceeded, bad credentials, unknown
# model, invalid parameters, ...) onto its own exception types. Retrying them only burns
# retries * retry_interval per failing sample, so pass them as no-retry exceptions.
# Transient errors (timeouts, connection failures, 429/5xx) stay retryable.
NON_RETRYABLE_LITELLM_ERRORS: Tuple[Type[Exception], ...] = (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)


def _track_provider_finish_reasons(response: Any) -> Optional[List[ChatCompletionChunk]]:
    """Keep original OpenAI terminal states before LiteLLM rewrites its chunks."""
    if not isinstance(response, litellm.CustomStreamWrapper):
        return None
    source = response.completion_stream
    finish_chunks: List[ChatCompletionChunk] = []

    def stream() -> Iterator[ChatCompletionChunk]:
        for chunk in source:
            if any(choice.finish_reason for choice in chunk.choices):
                finish_chunks.append(chunk.model_copy(deep=True))
            yield chunk

    async def async_stream() -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in source:
            if any(choice.finish_reason for choice in chunk.choices):
                finish_chunks.append(chunk.model_copy(deep=True))
            yield chunk

    if isinstance(source, Stream):
        response.completion_stream = stream()
    elif isinstance(source, AsyncStream):
        response.completion_stream = async_stream()
    else:
        return None
    return finish_chunks


def _validate_stream_finish_reason(response: Any) -> None:
    # CustomStreamWrapper synthesizes a terminal chunk even when the provider
    # never sent one. Check its recorded provider state rather than that chunk.
    if isinstance(response, litellm.CustomStreamWrapper) and not (
        response.received_finish_reason or response.intermittent_finish_reason
    ):
        raise RuntimeError('LiteLLM stream ended without a provider finish reason')


class LiteLLMAPI(ModelAPI):
    """LiteLLM model API provider.

    Uses litellm.completion() to route to any of 100+ LLM providers.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        request: Dict[str, Any] = {}

        messages = openai_chat_messages(
            input, reasoning_format=(config.reasoning_history or 'reasoning_field'), base_url=self.base_url
        )
        completion_params = openai_completion_params(
            model=self.model_name,
            config=config,
            tools=len(tools) > 0,
        )

        request = dict(
            # drop_params silently drops provider-unsupported kwargs
            # to prevent cross-provider errors
            drop_params=True,
            **completion_params,
        )
        request['messages'] = messages
        if len(tools) > 0:
            request['tools'] = openai_chat_tools(tools)
            request['tool_choice'] = openai_chat_tool_choice(tool_choice)
        if self.api_key:
            request['api_key'] = self.api_key
        if self.base_url:
            request['api_base'] = self.base_url

        try:
            t_start = time.monotonic()

            # Stream iteration is part of the attempt: discard partial chunks
            # and start a fresh request if reading the response fails.
            def _create_and_collect() -> Tuple[Any, Optional[float], bool]:
                response = litellm.completion(**request)
                if config.stream and not isinstance(response, ChatCompletion):
                    finish_chunks = _track_provider_finish_reasons(response)
                    completion, ttft = collect_stream_response(
                        response, request_start=t_start, finish_reason_chunks=finish_chunks
                    )
                    if finish_chunks is None:
                        _validate_stream_finish_reason(response)
                    return completion, ttft, True
                return response, None, False

            response, ttft, collected_stream = retry_call(
                _create_and_collect,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                no_retry_exceptions=NON_RETRYABLE_LITELLM_ERRORS,
            )

            completion = response if collected_stream else ChatCompletion(**response.model_dump())
            # Measure after the stream is consumed, as the openai_api path does.
            total_time = time.monotonic() - t_start

            choices = chat_choices_from_openai(completion, tools)
            output = model_output_from_openai(completion, choices)

            output.time = total_time
            usage = output.usage
            output.message.perf_metrics = PerformanceMetrics(
                latency=total_time,
                ttft=ttft,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            )
            return output

        except Exception as ex:
            logger.error(f'LiteLLM [{self.model_name}] error: {ex}')
            raise

    async def generate_async(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Async generation using litellm.acompletion().

        Uses ``litellm.acompletion()`` for non-blocking HTTP calls.  Request
        construction and response parsing are identical to the synchronous
        ``generate()``.
        """
        messages = openai_chat_messages(
            input, reasoning_format=(config.reasoning_history or 'reasoning_field'), base_url=self.base_url
        )
        completion_params = openai_completion_params(
            model=self.model_name,
            config=config,
            tools=len(tools) > 0,
        )

        request: Dict[str, Any] = dict(
            drop_params=True,
            **completion_params,
        )
        request['messages'] = messages
        if len(tools) > 0:
            request['tools'] = openai_chat_tools(tools)
            request['tool_choice'] = openai_chat_tool_choice(tool_choice)
        if self.api_key:
            request['api_key'] = self.api_key
        if self.base_url:
            request['api_base'] = self.base_url

        try:
            t_start = time.monotonic()

            async def _create_and_collect() -> Tuple[Any, Optional[float], bool]:
                response = await litellm.acompletion(**request)
                if config.stream and not isinstance(response, ChatCompletion):
                    finish_chunks = _track_provider_finish_reasons(response)
                    completion, ttft = await async_collect_stream_response(
                        response, request_start=t_start, finish_reason_chunks=finish_chunks
                    )
                    if finish_chunks is None:
                        _validate_stream_finish_reason(response)
                    return completion, ttft, True
                return response, None, False

            response, ttft, collected_stream = await async_retry_call(
                _create_and_collect,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                no_retry_exceptions=NON_RETRYABLE_LITELLM_ERRORS,
            )

            completion = response if collected_stream else ChatCompletion(**response.model_dump())
            # Measure after the stream is consumed, as the openai_api path does.
            total_time = time.monotonic() - t_start

            choices = chat_choices_from_openai(completion, tools)
            output = model_output_from_openai(completion, choices)

            output.time = total_time
            usage = output.usage
            output.message.perf_metrics = PerformanceMetrics(
                latency=total_time,
                ttft=ttft,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            )
            return output

        except Exception as ex:
            logger.error(f'LiteLLM [{self.model_name}] async error: {ex}')
            raise
