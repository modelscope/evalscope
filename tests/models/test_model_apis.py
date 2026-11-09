import sys
from types import ModuleType

import pytest

from evalscope.models import model_apis


def test_anthropic_api_rejects_older_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_apis, 'check_import', lambda *args, **kwargs: True)
    monkeypatch.setattr(model_apis.importlib.metadata, 'version', lambda package: '0.118.0')

    with pytest.raises(ImportError, match=r'anthropic>=0\.119\.0, found 0\.118\.0'):
        model_apis.anthropic_api()


@pytest.mark.parametrize('version', ['0.119.0', '0.120.0'])
def test_anthropic_api_accepts_supported_sdk(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setattr(model_apis, 'check_import', lambda *args, **kwargs: True)
    monkeypatch.setattr(model_apis.importlib.metadata, 'version', lambda package: version)
    module = ModuleType('evalscope.models.anthropic_compatible')
    api_class = type('AnthropicCompatibleAPI', (), {})
    module.AnthropicCompatibleAPI = api_class
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert model_apis.anthropic_api() is api_class
