"""System One transport for text single-choice decision models."""

import math
import os
import time
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.tool import ToolChoice, ToolInfo


class _ChoiceInput(BaseModel):
    """MCQ conversion data carried by an existing chat message's internal field."""

    model_config = ConfigDict(extra='forbid')

    state: Dict[str, JsonValue]
    instructions: str = Field(min_length=1)
    criteria: Dict[str, str] = Field(min_length=2, max_length=255)
    answer_prefix: Literal['ANSWER: ', '答案：'] = 'ANSWER: '

    @model_validator(mode='after')
    def validate_options(self) -> '_ChoiceInput':
        if any(not key.strip() or not value.strip() for key, value in self.criteria.items()):
            raise ValueError('Choice option keys and descriptions must not be empty.')
        return self


class _ChoiceAnswer(BaseModel):
    """Validated provider answer, retained in the raw response metadata."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['choice'] = 'choice'
    choice: str = Field(min_length=1)
    probabilities: Dict[str, float] = Field(min_length=2, max_length=255)
    confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode='after')
    def validate_distribution(self) -> '_ChoiceAnswer':
        values = self.probabilities.values()
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError('Choice probabilities must be finite values between 0 and 1.')
        # Providers may round each probability to two decimal places; retain those raw values.
        tolerance = max(0.01, len(self.probabilities) * 0.005) if all(round(v, 2) == v for v in values) else 0.01
        if not math.isclose(sum(values), 1, abs_tol=tolerance + 1e-12):
            raise ValueError('Choice probabilities do not sum to 1 within their rounding tolerance.')
        if self.choice not in self.probabilities:
            raise ValueError('The selected choice is missing from probabilities.')
        if self.probabilities[self.choice] < max(values) - 1e-6:
            raise ValueError('The selected choice must have the highest probability.')
        return self


class SystemOneAPI(ModelAPI):
    """Call a System One endpoint and retain the complete decision response."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, config, **kwargs)
        if not base_url or not base_url.rstrip('/').endswith('/v1'):
            raise ValueError('systemone_api requires an api_url ending in /v1.')
        if kwargs:
            raise ValueError(f'Unsupported System One model_args: {", ".join(sorted(kwargs))}.')
        self.validate_config(config)
        if api_key in (None, '', 'EMPTY'):
            endpoint_url = urlsplit(base_url)
            api_key = (
                os.getenv('TYPESAFE_API_KEY')
                if endpoint_url.scheme == 'https' and endpoint_url.hostname == 'api.typesafe.ai'
                else None
            )
            if api_key in ('', 'EMPTY'):
                api_key = None
        self.api_key = api_key
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self.client = httpx.Client(headers=headers, follow_redirects=False)
        self.endpoint = base_url.rstrip('/') + '/systemone'

    @staticmethod
    def validate_config(config: GenerateConfig) -> None:
        """Reject generation controls which have no System One equivalent."""
        supported = {'timeout', 'retries', 'retry_interval', 'batch_size'}
        unsupported = {
            key
            for key, value in config.model_dump(exclude_none=True, exclude_defaults=True).items()
            if key not in supported and not (key == 'stream' and value is False) and not (key == 'n' and value == 1)
        }
        if unsupported:
            raise ValueError(f'System One does not support generation_config: {", ".join(sorted(unsupported))}.')
        if config.retries is not None and config.retries < 1:
            raise ValueError('System One retries must be at least 1 (total attempts).')
        if config.retry_interval is not None and config.retry_interval < 0:
            raise ValueError('System One retry_interval must be non-negative.')

    def generate(
        self, input: List[ChatMessage], tools: List[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
    ) -> ModelOutput:
        """Convert an annotated MCQ chat request and return text for the existing answer extractor."""
        if tools:
            raise ValueError('System One Choice evaluation does not support tools.')
        self.validate_config(config)
        converted = self._convert_messages(input)
        system_prompt = '\n\n'.join(message.text for message in input if message.role == 'system' and message.text)
        instructions = f'{system_prompt}\n\n{converted.instructions}' if system_prompt else converted.instructions
        payload = {
            'model': self.model_name,
            'state': converted.state,
            'questions': {'answer': {'type': 'choice', 'instructions': instructions, 'criteria': converted.criteria}},
        }
        started = time.monotonic()
        attempts = config.retries if config.retries is not None else 5
        for attempt in range(attempts):
            try:
                response = self.client.post(self.endpoint, json=payload, timeout=config.timeout or 60)
            except httpx.TransportError:
                if attempt + 1 == attempts:
                    raise
            else:
                if response.status_code not in (429, 529) and response.status_code < 500:
                    response.raise_for_status()
                    break
                if attempt + 1 == attempts:
                    response.raise_for_status()
            time.sleep(config.retry_interval if config.retry_interval is not None else 10)

        raw = response.json()
        if not isinstance(raw, dict) or set(raw.get('answers', {})) != {'answer'}:
            raise ValueError('System One must return exactly the requested answer question.')
        result = _ChoiceAnswer.model_validate(raw['answers']['answer'])
        if set(result.probabilities) != set(converted.criteria):
            raise ValueError('Returned Choice options do not match the request criteria.')
        model = raw.get('model')
        if not isinstance(model, str) or not model:
            raise ValueError('System One response must include the resolved model name.')
        usage = raw.get('usage', {})
        output = ModelOutput.from_content(model=model, content=f'{converted.answer_prefix}{result.choice}')
        output.id = raw.get('request_id')
        output.usage = ModelUsage(
            input_tokens=usage.get('input_tokens', 0),
            output_tokens=usage.get('output_tokens', 0),
            total_tokens=usage.get('input_tokens', 0) + usage.get('output_tokens', 0),
        )
        output.time = time.monotonic() - started
        output.metadata = {
            'choice_request': payload,
            'choice_response': raw,
        }
        return output

    @staticmethod
    def _convert_messages(input: List[ChatMessage]) -> _ChoiceInput:
        """Read conversion data without attempting to parse arbitrary prompt text."""
        if any(message.role not in ('system', 'user') for message in input):
            raise ValueError('System One requires system messages and one annotated MCQ user message.')
        if any(
            not isinstance(message.content, str) and any(part.type != 'text' for part in message.content)
            for message in input
        ):
            raise ValueError('System One Choice supports text messages only.')
        users = [message for message in input if message.role == 'user']
        if len(users) != 1 or not isinstance(users[0].internal, dict) or 'systemone' not in users[0].internal:
            raise ValueError('System One requires MCQ conversion data in ChatMessage.internal["systemone"].')
        return _ChoiceInput.model_validate(users[0].internal['systemone'])

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        self.client.close()
