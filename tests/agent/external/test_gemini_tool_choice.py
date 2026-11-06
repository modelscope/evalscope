"""Preserve forced function selection through the Gemini bridge."""

import pytest

from evalscope.agent.external.bridge.translate_gemini import gemini_tool_choice
from evalscope.api.model import GenerateConfig, Model
from evalscope.api.tool import ToolFunction, ToolInfo
from evalscope.models.mockllm import MockLLM
from evalscope.models.utils.openai import openai_chat_tool_choice


@pytest.mark.parametrize('config_key', ['functionCallingConfig', 'function_calling_config'])
def test_any_single_allowed_function(config_key: str) -> None:
    choice = gemini_tool_choice({config_key: {'mode': 'ANY', 'allowedFunctionNames': ['get_weather']}})

    assert choice == ToolFunction(name='get_weather', arguments={})
    assert openai_chat_tool_choice(choice) == {'type': 'function', 'function': {'name': 'get_weather'}}


def test_single_allowed_function_filters_model_tools() -> None:
    choice = gemini_tool_choice({'functionCallingConfig': {'mode': 'ANY', 'allowedFunctionNames': ['get_weather']}})
    tools = [ToolInfo(name=name, description=name) for name in ['get_weather', 'get_time']]
    model = Model(api=MockLLM(model_name='mock-gemini'), config=GenerateConfig())

    _, filtered_tools, model_choice, _ = model._preprocess_input('What is the weather?', tools, choice)

    assert [tool.name for tool in filtered_tools] == ['get_weather']
    assert model_choice == ToolFunction(name='get_weather', arguments={})


@pytest.mark.parametrize(
    ('config', 'expected'),
    [
        (None, None),
        ({}, None),
        ({'functionCallingConfig': {}}, 'auto'),
        ({'functionCallingConfig': {'allowedFunctionNames': ['get_weather']}}, 'auto'),
        ({'functionCallingConfig': {'mode': 'ANY'}}, 'any'),
        ({'functionCallingConfig': {'mode': 'ANY', 'allowedFunctionNames': []}}, 'any'),
        ({'functionCallingConfig': {'mode': 'ANY', 'allowedFunctionNames': ['get_weather', 'get_time']}}, 'any'),
        ({'functionCallingConfig': {'mode': 'AUTO', 'allowedFunctionNames': ['get_weather']}}, 'auto'),
        ({'functionCallingConfig': {'mode': 'NONE', 'allowedFunctionNames': ['get_weather']}}, 'none'),
    ],
)
def test_other_tool_choices_unchanged(config: dict | None, expected: str | None) -> None:
    assert gemini_tool_choice(config) == expected
