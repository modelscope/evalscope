import os
import time
from openai import APIStatusError, BadRequestError, PermissionDeniedError, UnprocessableEntityError
from openai._types import NOT_GIVEN
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.messages import ChatMessage
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.argument_utils import get_supported_params
from evalscope.utils.function_utils import async_retry_call, retry_call
from .openai_compatible import OpenAICompatibleAPI
from .utils.openai import openai_handle_bad_request
from .utils.openai_responses import (
    async_collect_response_stream,
    chat_choices_from_openai_response,
    collect_response_stream,
    model_output_from_openai_response,
    openai_response_messages,
    openai_response_params,
    openai_response_tool_choice,
    openai_response_tools,
)

try:
    from openai.types.responses import Response
except ImportError as ex:
    Response = None
    _OPENAI_RESPONSES_IMPORT_ERROR: Optional[ImportError] = ex
else:
    _OPENAI_RESPONSES_IMPORT_ERROR = None

logger = get_logger()


class OpenAIResponsesAPI(OpenAICompatibleAPI):
    """OpenAI official Responses API provider."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        resolved_base_url = base_url or os.environ.get('EVALSCOPE_BASE_URL', None)
        if resolved_base_url is not None:
            resolved_base_url = resolved_base_url.rstrip('/').removesuffix('/responses')
        super().__init__(
            model_name=model_name,
            base_url=resolved_base_url,
            api_key=api_key,
            config=config,
            **model_args,
        )
        self._validate_responses_client()

    def _validate_responses_client(self) -> None:
        if _OPENAI_RESPONSES_IMPORT_ERROR is not None or not hasattr(self.client, 'responses'):
            raise RuntimeError(
                'The installed version of the "openai" library does not support the Responses API. '
                'Please upgrade to openai>=1.56.0.'
            )

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        request, tools, config = self._build_request(input, tools, tool_choice, config)

        try:
            t_start = time.monotonic()
            ttft: Optional[float] = None

            response = retry_call(
                self.client.responses.create,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                **request,
            )
            if not self._is_response_object(response):
                response, ttft = collect_response_stream(response, request_start=t_start)

            total_time = time.monotonic() - t_start
            return self._build_output(response, tools, total_time, ttft)

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
        request, tools, config = self._build_request(input, tools, tool_choice, config)

        try:
            t_start = time.monotonic()
            ttft: Optional[float] = None

            response = await async_retry_call(
                self.async_client.responses.create,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                **request,
            )
            if not self._is_response_object(response):
                response, ttft = await async_collect_response_stream(response, request_start=t_start)

            total_time = time.monotonic() - t_start
            return self._build_output(response, tools, total_time, ttft)

        except (BadRequestError, UnprocessableEntityError, PermissionDeniedError) as ex:
            return self.handle_bad_request(ex)
        except ValueError as ex:
            logger.error(f'Model [{self.model_name}] returned an invalid response: {ex}')
            raise

    def _build_request(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> Tuple[Dict[str, Any], List[ToolInfo], GenerateConfig]:
        tools, tool_choice, config = self.resolve_tools(tools, tool_choice, config)
        request = dict(
            input=openai_response_messages(input),
            tools=openai_response_tools(tools) if len(tools) > 0 else NOT_GIVEN,
            tool_choice=openai_response_tool_choice(tool_choice) if len(tools) > 0 else NOT_GIVEN,
            **self.response_params(config=config, tools=len(tools) > 0),
        )
        self.validate_request_params(request)
        return request, tools, config

    @staticmethod
    def _is_response_object(response: Any) -> bool:
        return Response is not None and isinstance(response, Response)

    def _build_output(
        self,
        response: Any,
        tools: List[ToolInfo],
        total_time: float,
        ttft: Optional[float],
    ) -> ModelOutput:
        self.on_response(response.model_dump())
        choices = self.chat_choices_from_response(response, tools)
        output = model_output_from_openai_response(response, choices)
        output.time = total_time
        usage = output.usage
        output.message.perf_metrics = PerformanceMetrics(
            latency=total_time,
            ttft=ttft,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
        )
        return output

    def response_params(self, config: GenerateConfig, tools: bool) -> Dict[str, Any]:
        return openai_response_params(
            model=self.model_name,
            config=config,
            tools=tools,
        )

    def validate_request_params(self, params: Dict[str, Any]) -> None:
        if not hasattr(self, '_valid_params'):
            self._valid_params = get_supported_params(self.client.responses.create)

        extra_body = params.get('extra_body', {})
        for key in list(params.keys()):
            if key not in self._valid_params:
                extra_body[key] = params.pop(key)

        if extra_body:
            params['extra_body'] = extra_body

    def chat_choices_from_response(self, response: Response, tools: List[ToolInfo]) -> List[ChatCompletionChoice]:
        return chat_choices_from_openai_response(response, tools)

    def handle_bad_request(self, ex: APIStatusError) -> Union[ModelOutput, Exception]:
        return openai_handle_bad_request(self.model_name, ex)
