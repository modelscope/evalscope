"""Unit tests for the three reasoning_format modes in openai_chat_message.

Each test pins the outgoing assistant-message dict shape so the breakage of
DeepSeek V4 thinking (issue #1392) cannot regress silently. The matrix covers
the three modes against representative ChatMessageAssistant shapes
(reasoning + text, tool_calls + reasoning, plain text, multiple reasoning
parts, blank reasoning).
"""

import pytest
from pydantic import ValidationError

from evalscope.api.messages import ChatMessageAssistant, ContentReasoning, ContentText
from evalscope.api.model import GenerateConfig
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.models.utils.openai import openai_chat_message, openai_completion_params
from evalscope.models.utils.openai_responses import openai_response_params


def _asst_reasoning_and_text(reasoning: str, text: str) -> ChatMessageAssistant:
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning), ContentText(text=text)],
    )


def _asst_tool_call_with_reasoning(reasoning: str) -> ChatMessageAssistant:
    tc = ToolCall(
        id='call-1',
        function=ToolFunction(name='lookup', arguments={'q': 'paris'}),
        type='function',
    )
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning)],
        tool_calls=[tc],
    )


# ---------------------------------------------------------------------------
# think_tag (default / pre-fix behavior)
# ---------------------------------------------------------------------------


def test_think_tag_embeds_reasoning_in_content():
    msg = _asst_reasoning_and_text('user wants paris weather', 'It is sunny.')
    out = openai_chat_message(msg, reasoning_format='think_tag')
    assert out['role'] == 'assistant'
    assert '<think>' in out['content']
    assert 'user wants paris weather' in out['content']
    assert 'It is sunny.' in out['content']
    assert 'reasoning_content' not in out


def test_think_tag_is_the_default():
    msg = _asst_reasoning_and_text('cot', 'reply')
    out = openai_chat_message(msg)
    assert '<think>' in out['content']
    assert 'reasoning_content' not in out


# ---------------------------------------------------------------------------
# reasoning_field (the fix)
# ---------------------------------------------------------------------------


def test_reasoning_field_promotes_to_top_level():
    msg = _asst_reasoning_and_text('user wants paris weather', 'It is sunny.')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert '<think>' not in out['content']
    assert 'It is sunny.' in out['content']
    assert out['reasoning_content'] == 'user wants paris weather'


def test_reasoning_field_with_tool_calls():
    msg = _asst_tool_call_with_reasoning('need to call lookup')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert out['reasoning_content'] == 'need to call lookup'
    assert '<think>' not in out['content']
    assert len(out['tool_calls']) == 1
    assert out['tool_calls'][0]['function']['name'] == 'lookup'


def test_reasoning_field_picks_last_when_multiple():
    msg = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning='first thought'),
            ContentText(text='intermediate'),
            ContentReasoning(reasoning='final thought'),
            ContentText(text='reply'),
        ],
    )
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert out['reasoning_content'] == 'final thought'


def test_reasoning_field_blank_falls_back_to_single_space():
    msg = _asst_reasoning_and_text('   ', 'reply')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert out['reasoning_content'] == ' '


def test_reasoning_field_no_reasoning_omits_key():
    msg = ChatMessageAssistant(content='plain reply, no reasoning')
    out = openai_chat_message(msg, reasoning_format='reasoning_field')
    assert 'reasoning_content' not in out
    assert out['content'] == 'plain reply, no reasoning'


# ---------------------------------------------------------------------------
# none (drop reasoning entirely, for DeepSeek R1 etc.)
# ---------------------------------------------------------------------------


def test_none_drops_reasoning_entirely():
    msg = _asst_reasoning_and_text('cot text', 'reply text')
    out = openai_chat_message(msg, reasoning_format='none')
    assert '<think>' not in out['content']
    assert 'cot text' not in out['content']
    assert 'reply text' in out['content']
    assert 'reasoning_content' not in out


def test_none_keeps_tool_calls():
    msg = _asst_tool_call_with_reasoning('cot')
    out = openai_chat_message(msg, reasoning_format='none')
    assert 'reasoning_content' not in out
    assert '<think>' not in out['content']
    assert len(out['tool_calls']) == 1


# ---------------------------------------------------------------------------
# GenerateConfig schema invariants (breaking change verification)
# ---------------------------------------------------------------------------


def test_reasoning_history_accepts_new_literal_values():
    for v in ('none', 'think_tag', 'reasoning_field'):
        cfg = GenerateConfig(reasoning_history=v)
        assert cfg.reasoning_history == v


def test_reasoning_history_rejects_legacy_literal_values():
    for v in ('all', 'last', 'auto'):
        with pytest.raises(ValidationError):
            GenerateConfig(reasoning_history=v)


def test_reasoning_history_default_is_none():
    assert GenerateConfig().reasoning_history is None


def test_reasoning_effort_passes_through_provider_specific_values():
    """Issue #1555: the effort whitelist blocked values that current providers accept."""
    for value in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
        params = openai_completion_params('gpt-5.1', GenerateConfig(reasoning_effort=value), tools=False)
        assert params['reasoning_effort'] == value


# ---------------------------------------------------------------------------
# OpenAI reasoning models on api.openai.com (issue #2003)
# ---------------------------------------------------------------------------

_OPENAI_BASE_URL = 'https://api.openai.com/v1'


def _sampling_config() -> GenerateConfig:
    return GenerateConfig(max_tokens=256, temperature=0.0, top_p=0.9)


@pytest.fixture
def warning_messages(monkeypatch):
    messages = []
    monkeypatch.setattr('evalscope.models.openai_compatible.logger.warning', messages.append)
    return messages


def test_reasoning_model_on_openai_endpoint_uses_max_completion_tokens():
    params = openai_completion_params('o3-mini', _sampling_config(), tools=False, base_url=_OPENAI_BASE_URL)
    assert params['max_completion_tokens'] == 256
    assert 'max_tokens' not in params


def test_reasoning_model_on_openai_endpoint_omits_sampling_params():
    params = openai_completion_params('gpt-5.1', _sampling_config(), tools=False, base_url=_OPENAI_BASE_URL)
    assert 'temperature' not in params
    assert 'top_p' not in params


def test_reasoning_model_on_other_endpoint_keeps_legacy_params():
    """A self-hosted OpenAI-compatible endpoint keeps the historical request shape."""
    params = openai_completion_params(
        'o3-mini', _sampling_config(), tools=False, base_url='https://vllm.internal/v1'
    )
    assert params['max_tokens'] == 256
    assert params['temperature'] == 0.0
    assert params['top_p'] == 0.9


def test_non_reasoning_model_on_openai_endpoint_keeps_sampling_params():
    params = openai_completion_params('gpt-4o', _sampling_config(), tools=False, base_url=_OPENAI_BASE_URL)
    assert params['max_tokens'] == 256
    assert params['temperature'] == 0.0
    assert params['top_p'] == 0.9


def test_lookalike_openai_host_is_not_treated_as_official():
    params = openai_completion_params(
        'o3-mini', _sampling_config(), tools=False, base_url='https://api.openai.com.example/v1'
    )
    assert params['max_tokens'] == 256
    assert params['temperature'] == 0.0


def test_responses_api_omits_sampling_params_for_openai_reasoning_models():
    params = openai_response_params('gpt-5.1', _sampling_config(), tools=False, base_url=_OPENAI_BASE_URL)
    assert params['max_output_tokens'] == 256
    assert 'temperature' not in params
    assert 'top_p' not in params


def test_responses_api_keeps_sampling_params_off_openai():
    params = openai_response_params(
        'gpt-5.1', _sampling_config(), tools=False, base_url='https://gateway.internal/v1'
    )
    assert params['temperature'] == 0.0
    assert params['top_p'] == 0.9


def test_model_api_builds_reasoning_safe_defaults_for_openai():
    """The model API forwards its base_url, so the default request reaches api.openai.com reasoning-safe."""
    from evalscope.models.openai_compatible import OpenAICompatibleAPI

    api = OpenAICompatibleAPI(model_name='o3-mini', base_url=_OPENAI_BASE_URL, api_key='sk-test')
    params = api.completion_params(_sampling_config(), tools=False)
    api.validate_request_params(params)
    assert params['max_completion_tokens'] == 256
    assert 'max_tokens' not in params
    assert 'temperature' not in params
    assert 'top_p' not in params


def test_model_api_warns_once_per_omitted_sampling_param(warning_messages):
    from evalscope.models.openai_compatible import OpenAICompatibleAPI

    api = OpenAICompatibleAPI(model_name='o3-mini', base_url=_OPENAI_BASE_URL, api_key='sk-test')
    api.completion_params(GenerateConfig(temperature=0.0), tools=False)
    api.completion_params(GenerateConfig(temperature=0.0), tools=False)
    api.completion_params(GenerateConfig(top_p=0.9), tools=False)

    assert sum('temperature' in message for message in warning_messages) == 1
    assert sum('top_p' in message for message in warning_messages) == 1


def test_responses_api_warns_once_for_omitted_sampling_params(warning_messages):
    from evalscope.models.openai_responses import OpenAIResponsesAPI

    api = OpenAIResponsesAPI(model_name='gpt-5.1', base_url=_OPENAI_BASE_URL, api_key='sk-test')
    api.response_params(_sampling_config(), tools=False)
    api.response_params(_sampling_config(), tools=False)

    assert len(warning_messages) == 1
    assert 'temperature' in warning_messages[0]
    assert 'top_p' in warning_messages[0]


def test_responses_api_builds_reasoning_safe_request():
    from evalscope.models.openai_responses import OpenAIResponsesAPI

    api = OpenAIResponsesAPI(model_name='gpt-5.1', base_url=_OPENAI_BASE_URL, api_key='sk-test')
    request, _, _ = api._build_request([], tools=[], tool_choice='auto', config=_sampling_config())
    assert request['max_output_tokens'] == 256
    assert 'max_tokens' not in request
    assert 'temperature' not in request
    assert 'top_p' not in request


@pytest.mark.parametrize('model,base_url', [('o3-mini', 'https://vllm.internal/v1'), ('gpt-4o', _OPENAI_BASE_URL)])
def test_model_api_does_not_warn_when_sampling_params_are_kept(warning_messages, model, base_url):
    from evalscope.models.openai_compatible import OpenAICompatibleAPI

    api = OpenAICompatibleAPI(model_name=model, base_url=base_url, api_key='sk-test')
    api.completion_params(_sampling_config(), tools=False)
    assert not warning_messages
