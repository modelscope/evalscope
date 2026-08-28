"""Regression tests for OpenAI-compatible audio request preprocessing."""

import asyncio
import base64
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import pytest
import requests
from openai.types.chat import ChatCompletion

from evalscope.api.messages import ChatMessageUser, ContentAudio
from evalscope.api.model import ChatCompletionChoice, GenerateConfig
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.models import openai_compatible
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.models.utils.openai import openai_chat_completion_part

DASHSCOPE_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
REMOTE_AUDIO_URL = 'https://example.test/audio'


def _mock_response(content: bytes) -> Mock:
    response = Mock()
    response.content = content
    return response


def _completion() -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'completion-id',
        'created': 1,
        'model': 'test-model',
        'object': 'chat.completion',
        'choices': [{
            'index': 0,
            'finish_reason': 'stop',
            'message': {
                'role': 'assistant',
                'content': 'done',
            },
        }],
    })


def _prepare_async_api(monkeypatch: pytest.MonkeyPatch, captured_request: Dict[str, Any]) -> OpenAICompatibleAPI:
    api = object.__new__(OpenAICompatibleAPI)
    api.base_url = 'https://example.test/v1'
    api.model_name = 'test-model'

    def resolve_tools(
        tools: List[ToolInfo], tool_choice: ToolChoice, config: GenerateConfig
    ) -> Tuple[List[ToolInfo], ToolChoice, GenerateConfig]:
        return tools, tool_choice, config

    def completion_params(config: GenerateConfig, tools: bool) -> Dict[str, Any]:
        return {'model': 'test-model'}

    def validate_request_params(request: Dict[str, Any]) -> None:
        return None

    def on_response(response: Dict[str, Any]) -> None:
        return None

    def chat_choices_from_completion(
        completion: ChatCompletion, tools: List[ToolInfo]
    ) -> List[ChatCompletionChoice]:
        return []

    def model_output_from_openai(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(usage=None, message=SimpleNamespace(), time=None)

    async def create(**request: Any) -> ChatCompletion:
        captured_request.update(request)
        return _completion()

    def async_client(_: OpenAICompatibleAPI) -> Any:
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    api.resolve_tools = resolve_tools
    api.completion_params = completion_params
    api.validate_request_params = validate_request_params
    api.on_response = on_response
    api.chat_choices_from_completion = chat_choices_from_completion
    monkeypatch.setattr(openai_compatible, 'model_output_from_openai', model_output_from_openai)
    monkeypatch.setattr(OpenAICompatibleAPI, 'async_client', property(async_client))
    return api


def test_http_audio_url_is_downloaded_and_sent_as_raw_base64() -> None:
    with patch('requests.get', return_value=_mock_response(b'WAVE')) as get:
        part = openai_chat_completion_part(ContentAudio(audio=REMOTE_AUDIO_URL, format='wav'))

    get.assert_called_once_with(REMOTE_AUDIO_URL, timeout=30)
    assert part['input_audio']['data'] == 'V0FWRQ=='
    assert base64.b64decode(part['input_audio']['data'], validate=True) == b'WAVE'


def test_http_audio_url_uses_audio_data_uri_for_dashscope() -> None:
    with patch('requests.get', return_value=_mock_response(b'WAVE')):
        part = openai_chat_completion_part(
            ContentAudio(audio=REMOTE_AUDIO_URL, format='wav'), base_url=DASHSCOPE_BASE_URL
        )

    assert part['input_audio']['data'] == 'data:audio/wav;base64,V0FWRQ=='


def test_extensionless_local_audio_uses_declared_mime_type(tmp_path: Path) -> None:
    audio_path = tmp_path / 'recording'
    audio_path.write_bytes(b'WAVE')

    part = openai_chat_completion_part(
        ContentAudio(audio=str(audio_path), format='mp3'), base_url=DASHSCOPE_BASE_URL
    )

    assert part['input_audio']['data'] == 'data:audio/mpeg;base64,V0FWRQ=='


def test_audio_data_uri_preserves_provider_compatibility() -> None:
    data_uri = 'data:audio/wav;base64,V0FWRQ=='

    dashscope_part = openai_chat_completion_part(
        ContentAudio(audio=data_uri, format='wav'), base_url=DASHSCOPE_BASE_URL
    )
    openai_part = openai_chat_completion_part(ContentAudio(audio=data_uri, format='wav'))

    assert dashscope_part['input_audio']['data'] == data_uri
    assert openai_part['input_audio']['data'] == 'V0FWRQ=='


def test_audio_download_failure_propagates() -> None:
    with patch('requests.get', side_effect=requests.Timeout('timed out')):
        with pytest.raises(requests.Timeout, match='timed out'):
            openai_chat_completion_part(ContentAudio(audio=REMOTE_AUDIO_URL, format='wav'))


def test_generate_async_keeps_event_loop_responsive_during_audio_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: Dict[str, Any] = {}
    api = _prepare_async_api(monkeypatch, captured_request)

    def delayed_get(*args: Any, **kwargs: Any) -> Mock:
        time.sleep(0.2)
        return _mock_response(b'WAVE')

    async def run() -> None:
        started = time.monotonic()
        generation = asyncio.create_task(
            api.generate_async(
                [ChatMessageUser(content=[ContentAudio(audio=REMOTE_AUDIO_URL, format='wav')])],
                [],
                None,
                GenerateConfig(retries=1),
            )
        )
        await asyncio.sleep(0.02)
        assert time.monotonic() - started < 0.1
        await generation

    with patch('requests.get', side_effect=delayed_get):
        asyncio.run(run())

    audio_data = captured_request['messages'][0]['content'][0]['input_audio']['data']
    assert audio_data == 'V0FWRQ=='
