import pytest

from evalscope.api.model import GenerateConfig


def _anthropic_utils():
    pytest.importorskip('anthropic')
    from evalscope.models.utils import anthropic

    return anthropic


def test_extra_headers_and_query_are_forwarded():
    # The OpenAI path forwards these; the Anthropic path used to drop them silently.
    anthropic = _anthropic_utils()

    params = anthropic.anthropic_completion_params(
        'claude',
        GenerateConfig(extra_headers={'X-Tenant': 'abc'}, extra_query={'api-version': '1'}),
    )

    assert params['extra_headers'] == {'X-Tenant': 'abc'}
    assert params['extra_query'] == {'api-version': '1'}


def test_extra_headers_and_query_absent_by_default():
    anthropic = _anthropic_utils()

    params = anthropic.anthropic_completion_params('claude', GenerateConfig())

    assert 'extra_headers' not in params
    assert 'extra_query' not in params
