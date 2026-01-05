import asyncio
from harbor.llms.base import BaseLLM, LLMResponse, UsageInfo
from pydantic import BaseModel, ConfigDict, PrivateAttr

from evalscope.api.messages.chat_message import dict_to_chat_message
from evalscope.api.model.model import Model
from evalscope.models.utils.openai import openai_chat_choices


class HarborLLM(BaseModel, BaseLLM):
    """A mock LLM that simulates sandboxed code execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _model: Model = PrivateAttr()

    def __init__(self, model: Model, **kwargs):
        super().__init__(**kwargs)
        self._model = model

    @property
    def model(self):
        return self._model

    async def call(self, prompt, **kwargs):
        message_history = kwargs.get('message_history', [])
        messages = message_history + [{'role': 'user', 'content': prompt}]

        # Run the blocking generate call in a separate thread
        loop = asyncio.get_running_loop()
        completion = await loop.run_in_executor(
            None, lambda: self._model.generate(input=[dict_to_chat_message(msg) for msg in messages])
        )

        # Process the completion to extract content and usage
        oa_choices = openai_chat_choices(completion.choices, include_reasoning=False)
        choice = oa_choices[0]
        msg = choice.message

        usage = completion.usage.model_dump(exclude_none=True)
        return LLMResponse(
            content=msg.content,
            usage=UsageInfo(
                prompt_tokens=usage.get('input_tokens', 0),
                completion_tokens=usage.get('output_tokens', 0),
                cache_tokens=usage.get('input_tokens_cache_read', 0),
                cost_usd=usage.get('cost_usd', 0.0)
            )
        )

    def get_model_context_limit(self):
        return self._model.config.max_tokens or 100_000
