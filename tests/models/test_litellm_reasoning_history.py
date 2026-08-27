from typing import Any, Dict, List

import litellm
from openai.types.chat import ChatCompletion

from evalscope.api.messages import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    ContentAudio,
    ContentReasoning,
    ContentText,
)
from evalscope.api.model import GenerateConfig
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.models.litellm_compatible import LiteLLMAPI


def _completion_response() -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion',
        'choices': [{
            'index': 0,
            'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': 'answer'},
        }],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    })


def _conversation() -> List[ChatMessage]:
    return [
        ChatMessageUser(content='question'),
        ChatMessageAssistant(content=[ContentReasoning(reasoning='prior thoughts'), ContentText(text='prior answer')]),
    ]


def _capturing_completion(calls: List[Dict[str, Any]]) -> Any:

    def _completion(**request: Any) -> ChatCompletion:
        calls.append(request)
        return _completion_response()

    return _completion


def test_reasoning_history_defaults_to_reasoning_field(monkeypatch: Any) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(litellm, 'completion', _capturing_completion(calls))

    api = LiteLLMAPI(model_name='openai/test-model')
    api.generate(input=_conversation(), tools=[], tool_choice='none', config=GenerateConfig())

    assistant_payload = calls[0]['messages'][1]
    # parity with OpenAICompatibleAPI: reasoning lives in the top-level field,
    # not smuggled into the content as a <think> tag
    assert assistant_payload['reasoning_content'] == 'prior thoughts'
    assert '<think>' not in assistant_payload['content']
    assert assistant_payload['content'].strip() == 'prior answer'


def test_reasoning_history_none_is_honored(monkeypatch: Any) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(litellm, 'completion', _capturing_completion(calls))

    api = LiteLLMAPI(model_name='openai/test-model')
    api.generate(
        input=_conversation(),
        tools=[],
        tool_choice='none',
        config=GenerateConfig(reasoning_history='none'),
    )

    assistant_payload = calls[0]['messages'][1]
    assert 'reasoning_content' not in assistant_payload
    assert '<think>' not in assistant_payload['content']
    assert assistant_payload['content'].strip() == 'prior answer'


def test_base_url_forwarded_for_dashscope_audio_encoding(monkeypatch: Any) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(litellm, 'completion', _capturing_completion(calls))

    api = LiteLLMAPI(model_name='openai/test-model', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
    api.generate(
        input=[
            ChatMessageUser(content=[
                ContentAudio(audio='data:audio/wav;base64,YXVkaW8=', format='wav'),
                ContentText(text='transcribe'),
            ])
        ],
        tools=[],
        tool_choice='none',
        config=GenerateConfig(),
    )

    audio_part = calls[0]['messages'][0]['content'][0]
    # DashScope endpoints require the data-URI prefix on input audio
    assert audio_part['input_audio']['data'] == 'data:audio/wav;base64,YXVkaW8='
