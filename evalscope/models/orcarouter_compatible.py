import os
from typing import Any, Optional

from evalscope.api.model import GenerateConfig, ModelAPI

from .openai_compatible import OpenAICompatibleAPI

# OrcaRouter is an OpenAI-compatible AI gateway (https://www.orcarouter.ai).
# Its default base URL exposes the same /v1 namespace as the OpenAI SDK, so the
# OpenAI-compatible provider can be reused as-is with a dedicated endpoint and
# API key.
ORCAROUTER_BASE_URL = 'https://api.orcarouter.ai/v1'


class OrcaRouterCompatibleAPI(OpenAICompatibleAPI):
    """OrcaRouter model API provider.

    Reuses the OpenAI-compatible chat path against OrcaRouter's gateway.
    The base URL defaults to ``https://api.orcarouter.ai/v1`` and the API key
    is read from the ``ORCAROUTER_API_KEY`` environment variable when not
    passed explicitly.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        resolved_base_url = base_url or os.environ.get('ORCAROUTER_BASE_URL', None) or ORCAROUTER_BASE_URL
        resolved_api_key = api_key or os.environ.get('ORCAROUTER_API_KEY', None)
        super().__init__(
            model_name=model_name,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            config=config,
            **model_args,
        )
