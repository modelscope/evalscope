import os
import time
from anthropic import Anthropic, APIStatusError, BadRequestError, PermissionDeniedError
from anthropic.types import Message
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.api.messages import ChatMessage
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.argument_utils import get_supported_params
from evalscope.utils.function_utils import retry_call
from .utils.anthropic import (
    anthropic_chat_messages,
    anthropic_chat_tool_choice,
    anthropic_chat_tools,
    anthropic_completion_params,
    anthropic_handle_bad_request,
    chat_choices_from_anthropic,
    collect_stream_response,
    model_output_from_anthropic,
)

logger = get_logger()


class AnthropicCompatibleAPI(ModelAPI):
    """Anthropic API compatible model implementation.

    This class provides a compatible interface for interacting with Anthropic's
    Claude models via their official API.
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

        # Use service prefix to lookup api_key
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY', None) or os.environ.get('EVALSCOPE_API_KEY', None)
        assert self.api_key, f'API key for {model_name} not found. Set ANTHROPIC_API_KEY or EVALSCOPE_API_KEY.'

        # Use service prefix to lookup base_url (optional for Anthropic)
        self.base_url = base_url or os.environ.get('ANTHROPIC_BASE_URL',
                                                   None) or os.environ.get('EVALSCOPE_BASE_URL', None)

        # Remove trailing slash from base_url if present
        if self.base_url:
            self.base_url = self.base_url.rstrip('/')
            # Anthropic SDK automatically appends /v1/messages, so we need to remove /v1 suffix
            # to avoid double /v1 in the URL (e.g., /v1/v1/messages)
            if self.base_url.endswith('/v1'):
                self.base_url = self.base_url[:-3]

        # Create Anthropic client
        client_kwargs: Dict[str, Any] = {
            'api_key': self.api_key,
            **model_args,
        }
        if self.base_url:
            client_kwargs['base_url'] = self.base_url

        self.client = Anthropic(**client_kwargs)

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response from the Anthropic API.

        Args:
            input: List of chat messages forming the conversation.
            tools: List of available tools for the model to use.
            tool_choice: How the model should choose which tool to use.
            config: Generation configuration parameters.

        Returns:
            ModelOutput containing the model's response.
        """
        # Setup request for logging/debugging
        request: Dict[str, Any] = {}
        response: Dict[str, Any] = {}

        tools, tool_choice, config = self.resolve_tools(tools, tool_choice, config)

        # Build completion parameters
        completion_params = self.completion_params(config)

        # Convert messages to Anthropic format
        system_message, messages = anthropic_chat_messages(input)

        # Build request
        request = dict(
            messages=messages,
            **completion_params,
        )

        # Add system message if present
        if system_message:
            request['system'] = system_message

        # Add tools if present
        if len(tools) > 0:
            request['tools'] = anthropic_chat_tools(tools)
            request['tool_choice'] = anthropic_chat_tool_choice(tool_choice)

        # Handle streaming
        if config.stream:
            request['stream'] = True

        self.validate_request_params(request)

        try:
            t_start = time.monotonic()
            ttft: Optional[float] = None

            # Generate completion
            message = retry_call(
                self.client.messages.create,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                **request,
            )

            # Handle streaming response
            if not isinstance(message, Message):
                message, ttft = collect_stream_response(message, request_start=t_start)

            total_time = time.monotonic() - t_start

            response = message.model_dump()
            self.on_response(response)

            # Build output and populate timing + perf metrics
            choices = self.chat_choices_from_message(message, tools)
            output = model_output_from_anthropic(message, choices)
            output.time = total_time
            usage = output.usage
            output.message.perf_metrics = PerformanceMetrics(
                latency=total_time,
                ttft=ttft,
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
            )
            return output

        except (BadRequestError, PermissionDeniedError) as ex:
            return self.handle_bad_request(ex)

    def resolve_tools(self, tools: List[ToolInfo], tool_choice: ToolChoice,
                      config: GenerateConfig) -> Tuple[List[ToolInfo], ToolChoice, GenerateConfig]:
        """Provides an opportunity for concrete classes to customize tool resolution."""
        return tools, tool_choice, config

    def completion_params(self, config: GenerateConfig) -> Dict[str, Any]:
        """Build Anthropic completion parameters from config."""
        return anthropic_completion_params(
            model=self.model_name,
            config=config,
        )

    def validate_request_params(self, params: Dict[str, Any]):
        """Hook for subclasses to do custom request parameter validation."""
        # Cache supported params to avoid repeated calls to inspect.signature.
        if not hasattr(self, '_valid_params'):
            self._valid_params = get_supported_params(self.client.messages.create)

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

    def chat_choices_from_message(self, message: Message, tools: List[ToolInfo]) -> List[ChatCompletionChoice]:
        """Hook for subclasses to do custom chat choice processing."""
        return chat_choices_from_anthropic(message, tools)

    def handle_bad_request(self, ex: APIStatusError) -> Union[ModelOutput, Exception]:
        """Hook for subclasses to do bad request handling."""
        result = anthropic_handle_bad_request(self.model_name, ex)
        if isinstance(result, Exception):
            raise result
        return result
