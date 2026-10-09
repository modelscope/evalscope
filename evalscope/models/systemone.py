"""System One transport for text single-choice decision models."""

import os
import time
from typing import Any, List, Optional
from urllib.parse import urlsplit

import httpx

from evalscope.api.messages import ChatMessage
from evalscope.api.model import ChoiceRequest, ChoiceResult, GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.api.tool import ToolChoice, ToolInfo


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
        """Reject chat generation; audited adapters must call generate_choice()."""
        raise NotImplementedError('systemone_api requires a structured Choice request from a supported benchmark.')

    def generate_choice(self, request: ChoiceRequest, config: GenerateConfig) -> ModelOutput:
        """Run a single structured decision, retrying only transient transport failures."""
        self.validate_config(config)
        payload = request.to_payload(self.model_name)
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
        result = ChoiceResult.model_validate(raw['answers']['answer'])
        result.validate_request(request)
        model = raw.get('model')
        if not isinstance(model, str) or not model:
            raise ValueError('System One response must include the resolved model name.')
        usage = raw.get('usage', {})
        output = ModelOutput.from_content(model=model, content=result.choice)
        output.choice_result = result
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

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        self.client.close()
