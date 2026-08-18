import pytest

from evalscope.benchmarks.bfcl.v4.bfcl_v4_adapter import _normalize_openai_base_url
from evalscope.benchmarks.bfcl.v4.utils import patch_openai_completions_handler_empty_tool_calls


@pytest.mark.parametrize(
    ('api_url', 'expected'),
    [
        ('https://example.test/v1', 'https://example.test/v1'),
        ('https://example.test/v1/', 'https://example.test/v1'),
        ('https://example.test/v1/chat/completions', 'https://example.test/v1'),
        ('https://example.test/v1/chat/completions/', 'https://example.test/v1'),
        ('  https://example.test/v1/chat/completions/  ', 'https://example.test/v1'),
        ('', ''),
        ('   ', ''),
        (None, ''),
    ],
)
def test_normalize_openai_base_url(api_url, expected):
    assert _normalize_openai_base_url(api_url) == expected


class _FakeMessage:

    def __init__(self, tool_calls, content):
        self.tool_calls = tool_calls
        self.content = content


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeResponse:

    def __init__(self, tool_calls, content):
        self.choices = [type('Choice', (), {'message': _FakeMessage(tool_calls, content)})()]
        self.usage = _FakeUsage()


class _FakeToolCall:

    def __init__(self, id_, name, arguments):
        self.id = id_
        self.function = type('Function', (), {'name': name, 'arguments': arguments})()


class _FakeHandler:
    """Stand-in for bfcl_eval's OpenAICompletionsHandler; avoids a hard test dependency
    on the (optional) ``bfcl_eval`` package being installed."""
    pass


def test_patch_empty_tool_calls_falls_back_to_content():
    # Mirrors what vLLM/SGLang-style OpenAI-compatible servers return for a
    # text-only assistant turn: `tool_calls: []` instead of `None`.
    patch_openai_completions_handler_empty_tool_calls(_FakeHandler)
    handler = _FakeHandler()

    parsed = handler._parse_query_response_FC(_FakeResponse(tool_calls=[], content='The answer is 42.'))
    assert parsed['model_responses'] == 'The answer is 42.'
    assert parsed['tool_call_ids'] == []


def test_patch_none_tool_calls_still_falls_back_to_content():
    patch_openai_completions_handler_empty_tool_calls(_FakeHandler)
    handler = _FakeHandler()

    parsed = handler._parse_query_response_FC(_FakeResponse(tool_calls=None, content='hello'))
    assert parsed['model_responses'] == 'hello'
    assert parsed['tool_call_ids'] == []


def test_patch_real_tool_calls_still_parsed_normally():
    patch_openai_completions_handler_empty_tool_calls(_FakeHandler)
    handler = _FakeHandler()

    calls = [_FakeToolCall('call_1', 'core_memory_add', '{"key": "a", "value": "b"}')]
    parsed = handler._parse_query_response_FC(_FakeResponse(tool_calls=calls, content=None))
    assert parsed['model_responses'] == [{'core_memory_add': '{"key": "a", "value": "b"}'}]
    assert parsed['tool_call_ids'] == ['call_1']


def test_patch_against_real_bfcl_eval_handler():
    """Same behavior as the fake-stand-in tests above, but exercised against the
    actual bfcl_eval.OpenAICompletionsHandler class and real openai response
    types, so the patch is verified against the thing it's actually patching."""
    pytest.importorskip('bfcl_eval')
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
    from openai.types.completion_usage import CompletionUsage

    def _make_response(tool_calls, content):
        message = ChatCompletionMessage(role='assistant', content=content, tool_calls=tool_calls)
        return ChatCompletion(
            id='chatcmpl-test',
            choices=[Choice(finish_reason='stop', index=0, message=message)],
            created=0,
            model='test-model',
            object='chat.completion',
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    # Uninitialized instance: `_parse_query_response_FC` only touches its
    # `api_response` argument, so a real handler's `__init__` isn't needed.
    handler = object.__new__(OpenAICompletionsHandler)

    # Before patching: the upstream bug drops the text-only answer when the
    # server returns `tool_calls: []` instead of omitting/nulling it.
    unpatched_response = _make_response(tool_calls=[], content='The answer is 42.')
    parsed = handler._parse_query_response_FC(unpatched_response)
    assert parsed['model_responses'] == []

    patch_openai_completions_handler_empty_tool_calls(OpenAICompletionsHandler)

    parsed = handler._parse_query_response_FC(unpatched_response)
    assert parsed['model_responses'] == 'The answer is 42.'
    assert parsed['tool_call_ids'] == []

    tool_call = ChatCompletionMessageToolCall(
        id='call_1',
        type='function',
        function=Function(name='core_memory_add', arguments='{"key": "a", "value": "b"}'),
    )
    real_tool_call_response = _make_response(tool_calls=[tool_call], content=None)
    parsed = handler._parse_query_response_FC(real_tool_call_response)
    assert parsed['model_responses'] == [{'core_memory_add': '{"key": "a", "value": "b"}'}]
    assert parsed['tool_call_ids'] == ['call_1']
