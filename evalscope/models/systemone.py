"""Shared transport for single-question text decision APIs."""

import os
import time
from typing import Any, List, Literal, Optional
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.model.decision import DecisionAnswer, DecisionInput, parse_decision_response, parse_decision_usage
from evalscope.api.tool import ToolChoice, ToolInfo


class _ChoiceAnswer(DecisionAnswer):
    """Compatibility contract for saved Choice responses."""

    type: Literal['choice'] = 'choice'
    choice: str = Field(min_length=1)


class SystemOneAPI(ModelAPI):
    """Call a System One endpoint and retain the complete decision response."""

    protocol = 'systemone'

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
                if self.protocol == 'systemone'
                and endpoint_url.scheme == 'https'
                and endpoint_url.hostname == 'api.typesafe.ai'
                else None
            )
            if api_key in ('', 'EMPTY'):
                api_key = None
        self.api_key = api_key
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self.client = httpx.Client(headers=headers, follow_redirects=False)
        self.endpoint = base_url.rstrip('/') + ('/systemone' if self.protocol == 'systemone' else '/decide')

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
        """Send an annotated decision request and retain its structured result."""
        if tools:
            raise ValueError('Decision APIs do not support tools.')
        self.validate_config(config)
        converted = self._convert_messages(input)
        system_prompt = '\n\n'.join(message.text for message in input if message.role == 'system' and message.text)
        instructions = f'{system_prompt}\n\n{converted.instructions}' if system_prompt else converted.instructions
        converted = converted.model_copy(update={'instructions': instructions})
        payload = (
            converted.systemone_payload(self.model_name)
            if self.protocol == 'systemone'
            else converted.autotrust_payload()
        )
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
        result = parse_decision_response(raw, converted, self.protocol)
        model = raw['model']
        input_tokens, output_tokens = parse_decision_usage(raw)
        value = (
            result.choice
            if result.type == 'choice'
            else ('yes' if result.noul >= 0.5 else 'no')
            if result.type == 'noul'
            else str(result.score)
        )
        output = ModelOutput.from_content(model=model, content=f'{converted.answer_prefix}{value}')
        output.id = raw.get('request_id')
        output.usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        output.time = time.monotonic() - started
        output.metadata = {
            'choice_request': payload,
            'choice_response': raw,
            'decision': result.model_dump(exclude_none=True),
            'decision_request': converted.model_dump(),
            'transport_attempts': attempt + 1,
            'model_revision': raw.get('model_revision'),
        }
        return output

    @staticmethod
    def _convert_messages(input: List[ChatMessage]) -> DecisionInput:
        """Read conversion data without attempting to parse arbitrary prompt text."""
        if any(message.role not in ('system', 'user') for message in input):
            raise ValueError('Decision APIs require system messages and one annotated user message.')
        if any(
            not isinstance(message.content, str) and any(part.type != 'text' for part in message.content)
            for message in input
        ):
            raise ValueError('Decision APIs support text messages only.')
        users = [message for message in input if message.role == 'user']
        if len(users) != 1 or not isinstance(users[0].internal, dict) or 'systemone' not in users[0].internal:
            raise ValueError('Decision APIs require conversion data in ChatMessage.internal["systemone"].')
        return DecisionInput.model_validate(users[0].internal['systemone'])

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        self.client.close()


class AutoTrustAPI(SystemOneAPI):
    """Serve native AutoTrust JEV decisions through /v1/decide."""

    protocol = 'autotrust'
