"""LiteLLM model API provider for EvalScope.

Routes to 100+ LLM providers via litellm.completion().
Provider API keys are read from environment variables automatically
(OPENAI_API_KEY, ANTHROPIC_API_KEY, AWS_ACCESS_KEY_ID, GEMINI_API_KEY, etc.).

Model names use LiteLLM format: "provider/model-name".
See https://docs.litellm.ai/docs/providers for the full list.
"""

import time
from openai.types.chat import ChatCompletion
from typing import Any, Dict, List, Optional, Tuple

from evalscope.api.messages import ChatMessage
from evalscope.api.messages.perf_metrics import PerformanceMetrics
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils import get_logger
from evalscope.utils.function_utils import retry_call
from .utils.openai import (
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
        import litellm

        request: Dict[str, Any] = {}

        messages = openai_chat_messages(input)
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

            response = retry_call(
                litellm.completion,
                retries=config.retries,
                sleep_interval=config.retry_interval,
                **request,
            )

            total_time = time.monotonic() - t_start
            ttft: Optional[float] = None

            if config.stream and not isinstance(response, ChatCompletion):
                completion, ttft = collect_stream_response(response, request_start=t_start)
            else:
                completion = ChatCompletion(**response.model_dump())

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
