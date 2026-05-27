import base64
from typing import Any

from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.model import get_model
from evalscope.benchmarks.seed_tts_eval.seed_tts_eval_adapter import PROMPT_TEMPLATE
from evalscope.constants import EvalType
from evalscope.utils.url_utils import data_uri_to_base64


def test_text2speech_volcengine_provider_returns_audio(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    class MockRequest:

        headers = {'X-Api-Request-Id': 'request-1'}

    class MockResponse:

        headers = {'X-Tt-Logid': 'log-1'}
        request = MockRequest()
        text = ''

        def __init__(self, body: str):
            self.body = body

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int, decode_unicode: bool) -> list[str]:
            return [self.body]

    def mock_post(
        self: Any,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        stream: bool,
        timeout: float,
    ) -> MockResponse:
        calls['url'] = url
        calls['headers'] = headers
        calls['json'] = json
        calls['stream'] = stream
        calls['timeout'] = timeout
        audio = base64.b64encode(b'audio-bytes').decode('utf-8')
        body = ''.join([
            '{"code":0,"message":"","data":"',
            audio,
            '"}',
            '{"code":20000000,"message":"OK","data":null,"usage":{"text_words":5}}',
        ])
        return MockResponse(body)

    monkeypatch.setattr('requests.Session.post', mock_post)
    model = get_model(
        model='seed-tts-2.0',
        eval_type=EvalType.TEXT2SPEECH,
        base_url='https://openspeech.test/api/v3/tts/unidirectional',
        api_key='test-key',
        model_args={
            'speaker': 'en_female_dacey_uranus_bigtts',
            'audio_params': {
                'sample_rate': 24000,
            },
        },
        memoize=False,
    )
    output = model.generate([
        ChatMessageUser(
            content=[
                ContentAudio(audio='data:audio/wav;base64,YXVkaW8=', format='wav'),
                ContentText(
                    text=PROMPT_TEMPLATE.format(
                        prompt_text='reference speech',
                        text='Please synthesize this sentence.',
                    )
                ),
            ]
        )
    ])

    assert calls['url'] == 'https://openspeech.test/api/v3/tts/unidirectional'
    assert calls['headers']['X-Api-Key'] == 'test-key'
    assert calls['headers']['X-Api-Resource-Id'] == 'seed-tts-2.0'
    assert calls['json']['req_params']['text'] == 'Please synthesize this sentence.'
    assert calls['json']['req_params']['speaker'] == 'en_female_dacey_uranus_bigtts'
    assert calls['json']['req_params']['audio_params'] == {
        'format': 'mp3',
        'sample_rate': 24000,
    }
    assert calls['stream'] is True
    audio_content = output.message.content[0]
    assert isinstance(audio_content, ContentAudio)
    assert audio_content.format == 'mp3'
    assert base64.b64decode(data_uri_to_base64(audio_content.audio)) == b'audio-bytes'
    assert output.metadata['usage'] == {'text_words': 5}
