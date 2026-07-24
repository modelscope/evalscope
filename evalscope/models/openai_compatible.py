import os
import time
from openai import APIStatusError, AsyncOpenAI, BadRequestError, OpenAI, PermissionDeniedError, UnprocessableEntityError
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletion
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.messages import ChatMessage
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.argument_utils import get_supported_params
from evalscope.utils.function_utils import async_retry_call, retry_call
from .utils.async_client import LoopBoundAsyncClientPool
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


class OpenAICompatibleAPI(ModelAPI):

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

        # use service prefix to lookup api_key
        self.api_key = api_key or os.environ.get('EVALSCOPE_API_KEY', None)
        assert self.api_key, f'API key for {model_name} not found'

        # use service prefix to lookup base_url
        self.base_url = base_url or os.environ.get('EVALSCOPE_BASE_URL', None)
        assert self.base_url, f'Base URL for {model_name} not found'

        # remove trailing slash from base_url
        self.base_url = self.base_url.rstrip('/').removesuffix('/chat/completions')

        # create http client (sync; reused across threads)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **model_args,
        )

        # AsyncOpenAI wraps loop-bound httpx/anyio resources. Keep one client
        # per loop so worker-owned and caller-managed loops remain isolated.
        self._async_client_kwargs: Dict[str, Any] = {
            'api_key': self.api_key,
            'base_url': self.base_url,
            **model_args,
        }
        self._async_client_pool = LoopBoundAsyncClientPool(lambda: AsyncOpenAI(**self._async_client_kwargs))

    @property
    def async_client(self) -> AsyncOpenAI:
        """Return an AsyncOpenAI bound to the currently running event loop.

        Lazily constructed once per loop and cached. Callers must invoke
        from within a coroutine running on a real event loop.
        """
        return self._async_client_pool.get()

    async def aclose(self) -> None:
        """Close all loop-bound async clients while keeping this model reusable."""
        await self._async_client_pool.aclose()

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # setup request and response for ModelCall
        request: Dict[str, Any] = {}
        response: Dict[str, Any] = {}

        tools, tool_choice, config = self.resolve_tools(tools, tool_choice, config)

        # get completion params (slice off service from model name)
        completion_params = self.completion_params(
            config=config,
            tools=len(tools) > 0,
        )

        request = dict(
            messages=openai_chat_messages(
                input, reasoning_format=(config.reasoning_history or 'reasoning_field'), base_url=self.base_url
            ),
            tools=openai_chat_tools(tools) if len(tools) > 0 else NOT_GIVEN,
            tool_choice=openai_chat_tool_choice(tool_choice) if len(tools) > 0 else NOT_GIVEN,
            **completion_params,
        )

        self.validate_request_params(request)

        try:
            t_start = time.monotonic()
            ttft: Optional[float] = None

            # A streaming request is not complete when create() returns: the
            # connection may still fail while its chunks are being consumed.
            # Retry the whole request so a partial response is discarded and
            # replaced by one complete response.
            def _create_and_collect() -> Tuple[ChatCompletion, Optional[float]]:
                raw_completion = self.client.chat.completions.create(**request)
                if isinstance(raw_completion, ChatCompletion):
                    return raw_completion, None
                return collect_stream_response(raw_completion, request_start=t_start)

            completion, ttft = retry_call(
                _create_and_collect,
                retries=config.retries,
                sleep_interval=config.retry_interval,
            )

            total_time = time.monotonic() - t_start

            response = completion.model_dump()
            self.on_response(response)

            # return output and call
            choices = self.chat_choices_from_completion(completion, tools)
            output = model_output_from_openai(completion, choices)

            # Populate timing fields
            output.time = total_time
            usage = output.usage
            output.message.perf_metrics = PerformanceMetrics(
                latency=total_time,
                ttft=ttft,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            )
            return output

        except (BadRequestError, UnprocessableEntityError, PermissionDeniedError) as ex:
            return self.handle_bad_request(ex)
        except ValueError as ex:
            logger.error(f'Model [{self.model_name}] returned an invalid response: {ex}')
            raise

    async def generate_async(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Native async generation using AsyncOpenAI client.

        Uses ``self.async_client`` for non-blocking HTTP calls, keeping the
        event loop free for concurrent tasks.  Request construction and
        response parsing are identical to the synchronous ``generate()``.
        """
        request: Dict[str, Any] = {}

        tools, tool_choice, config = self.resolve_tools(tools, tool_choice, config)

        completion_params = self.completion_params(
            config=config,
            tools=len(tools) > 0,
        )

        request = dict(
            messages=openai_chat_messages(
                input, reasoning_format=(config.reasoning_history or 'reasoning_field'), base_url=self.base_url
            ),
            tools=openai_chat_tools(tools) if len(tools) > 0 else NOT_GIVEN,
            tool_choice=openai_chat_tool_choice(tool_choice) if len(tools) > 0 else NOT_GIVEN,
            **completion_params,
        )

        self.validate_request_params(request)

        try:
            t_start = time.monotonic()
            ttft: Optional[float] = None

            # Keep stream consumption inside the retry boundary. If an async
            # stream is interrupted, start a fresh request rather than
            # returning or persisting its partial response.
            async def _create_and_collect() -> Tuple[ChatCompletion, Optional[float]]:
                raw_completion = await self.async_client.chat.completions.create(**request)
                if isinstance(raw_completion, ChatCompletion):
                    return raw_completion, None
                return await async_collect_stream_response(raw_completion, request_start=t_start)

            completion, ttft = await async_retry_call(
                _create_and_collect,
                retries=config.retries,
                sleep_interval=config.retry_interval,
            )

            total_time = time.monotonic() - t_start

            response = completion.model_dump()
            self.on_response(response)

            # Return output
            choices = self.chat_choices_from_completion(completion, tools)
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

        except (BadRequestError, UnprocessableEntityError, PermissionDeniedError) as ex:
            return self.handle_bad_request(ex)
        except ValueError as ex:
            logger.error(f'Model [{self.model_name}] returned an invalid response: {ex}')
            raise

    def resolve_tools(self, tools: List[ToolInfo], tool_choice: ToolChoice,
                      config: GenerateConfig) -> Tuple[List[ToolInfo], ToolChoice, GenerateConfig]:
        """Provides an opportunity for concrete classes to customize tool resolution."""
        return tools, tool_choice, config

    def completion_params(self, config: GenerateConfig, tools: bool) -> Dict[str, Any]:
        return openai_completion_params(
            model=self.model_name,
            config=config,
            tools=tools,
        )

    def validate_request_params(self, params: Dict[str, Any]):
        """Hook for subclasses to do custom request parameter validation."""
        # Cache supported params to avoid repeated calls to inspect.signature.
        if not hasattr(self, '_valid_params'):
            self._valid_params = get_supported_params(self.client.chat.completions.create)

        # Move unsupported parameters to extra_body.
        extra_body = params.get('extra_body', {})
        for key in list(params.keys()):
            if key not in self._valid_params:
                extra_body[key] = params.pop(key)

        if extra_body:
            params['extra_body'] = extra_body

    def on_response(self, response: Dict[str, Any]) -> None:
        """Hook for subclasses to do custom response handling."""
        pass

    def chat_choices_from_completion(self, completion: ChatCompletion,
                                     tools: List[ToolInfo]) -> List[ChatCompletionChoice]:
        """Hook for subclasses to do custom chat choice processing."""
        return chat_choices_from_openai(completion, tools)

    def handle_bad_request(self, ex: APIStatusError) -> Union[ModelOutput, Exception]:
        """Hook for subclasses to do bad request handling"""
        return openai_handle_bad_request(self.model_name, ex)
