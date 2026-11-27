from typing import Any, Dict, List

import pytest

from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.models import modelscope as modelscope_module
from evalscope.models.modelscope import GenerateOutput, ModelScopeAPI
from tests.models.test_modelscope_usage import _make_api


def _generate_kwargs(monkeypatch: Any, config: GenerateConfig) -> Dict[str, Any]:
    """Return the keyword arguments ``ModelScopeAPI.generate`` passes to ``model.generate``."""
    captured: List[Any] = []

    def fake_batched_generate(generate_input: Any) -> List[GenerateOutput]:
        captured.append(generate_input)
        return [
            GenerateOutput(
                output='ANSWER: A',
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                logprobs=None,
                time=0.0,
                stop_reason='stop',
            )
        ]

    monkeypatch.setattr(modelscope_module, 'batched_generate', fake_batched_generate)
    api: ModelScopeAPI = _make_api()
    api.generate(input=[ChatMessageUser(content='hi')], tools=[], tool_choice='none', config=config)
    return captured[0].generator.keywords


def test_temperature_zero_decodes_greedily(monkeypatch: Any) -> None:
    # Transformers rejects temperature=0 when the model's generation config samples (Qwen, Llama instruct).
    kwargs = _generate_kwargs(monkeypatch, GenerateConfig(temperature=0.0))
    assert kwargs['do_sample'] is False
    assert 'temperature' not in kwargs


def test_temperature_zero_with_do_sample_false(monkeypatch: Any) -> None:
    kwargs = _generate_kwargs(monkeypatch, GenerateConfig(temperature=0.0, do_sample=False))
    assert kwargs['do_sample'] is False
    assert 'temperature' not in kwargs


@pytest.mark.parametrize('do_sample', [None, True])
def test_positive_temperature_is_passed_through(monkeypatch: Any, do_sample: Any) -> None:
    kwargs = _generate_kwargs(monkeypatch, GenerateConfig(temperature=0.6, do_sample=do_sample))
    assert kwargs['temperature'] == 0.6
    assert kwargs.get('do_sample') is do_sample
