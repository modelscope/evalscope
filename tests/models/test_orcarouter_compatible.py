from typing import Any

import pytest

from evalscope.api.registry import get_model_api
from evalscope.constants import EvalType


def test_orcarouter_eval_type_registered() -> None:
    """orcarouter_api must be a registered model API and an EvalType."""
    assert EvalType.ORCAROUTER_API == 'orcarouter_api'
    api_cls = get_model_api('orcarouter_api')
    assert api_cls is not None


def test_orcarouter_defaults_to_gateway_base_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider defaults to the OrcaRouter gateway URL and ORCAROUTER_API_KEY."""
    from evalscope.models import openai_compatible
    from evalscope.models.orcarouter_compatible import ORCAROUTER_BASE_URL, OrcaRouterCompatibleAPI

    monkeypatch.setattr(openai_compatible, 'OpenAI', lambda **kwargs: object())
    created: list[Any] = []
    monkeypatch.setattr(
        openai_compatible,
        'AsyncOpenAI',
        lambda **kwargs: created.append(kwargs) or object(),
    )
    monkeypatch.setenv('ORCAROUTER_API_KEY', 'sk-orca-test')

    api = OrcaRouterCompatibleAPI(model_name='orcarouter/free')
    assert api.base_url == ORCAROUTER_BASE_URL
    assert api.api_key == 'sk-orca-test'


def test_orcarouter_accepts_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit base_url / api_key takes precedence over defaults."""
    from evalscope.models import openai_compatible
    from evalscope.models.orcarouter_compatible import OrcaRouterCompatibleAPI

    monkeypatch.setattr(openai_compatible, 'OpenAI', lambda **kwargs: object())
    monkeypatch.setattr(openai_compatible, 'AsyncOpenAI', lambda **kwargs: object())
    monkeypatch.setenv('ORCAROUTER_API_KEY', 'sk-orca-default')

    api = OrcaRouterCompatibleAPI(
        model_name='orcarouter/fusion',
        base_url='https://custom.example/v1',
        api_key='sk-orca-custom',
    )
    assert api.base_url == 'https://custom.example/v1'
    assert api.api_key == 'sk-orca-custom'
