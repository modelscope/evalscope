import pytest

from evalscope.api.model import GenerateConfig


def _anthropic_utils():
    pytest.importorskip('anthropic')
    from evalscope.models.utils import anthropic

    return anthropic


def test_thinking_omits_temperature_and_top_k():
    # The Anthropic API rejects temperature != 1 and any top_k when extended thinking is enabled.
    anthropic = _anthropic_utils()

    params = anthropic.anthropic_completion_params(
        'claude', GenerateConfig(temperature=0.0, top_k=40, reasoning_tokens=2048)
    )

    assert params['thinking'] == {'type': 'enabled', 'budget_tokens': 2048}
    assert 'temperature' not in params
    assert 'top_k' not in params


def test_without_thinking_keeps_temperature_and_top_k():
    anthropic = _anthropic_utils()

    params = anthropic.anthropic_completion_params('claude', GenerateConfig(temperature=0.0, top_k=40))

    assert params['temperature'] == 0.0
    assert params['top_k'] == 40
    assert 'thinking' not in params
