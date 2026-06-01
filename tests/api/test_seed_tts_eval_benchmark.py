import base64
import pytest
from typing import Any, Optional

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.model import ChatCompletionChoice, ModelOutput
from evalscope.benchmarks.seed_tts_eval.seed_tts_eval_adapter import PROMPT_TEMPLATE, SeedTTSEvalAdapter
from evalscope.config import TaskConfig
from evalscope.metrics.metric import AudioWER


def make_adapter(metric_list: Optional[list[Any]] = None) -> SeedTTSEvalAdapter:
    meta = BenchmarkMeta(
        name='seed_tts_eval',
        dataset_id='evalscope/Seed-TTS-Eval',
        subset_list=['en'],
        default_subset='en',
        eval_split='train',
        prompt_template=PROMPT_TEMPLATE,
        metric_list=metric_list or [],
    )
    cfg = TaskConfig(datasets=['seed_tts_eval'])
    adapter = SeedTTSEvalAdapter(benchmark_meta=meta, task_config=cfg)
    adapter.current_subset_name = 'en'
    return adapter


def test_record_to_sample_builds_tts_prompt_with_reference_audio() -> None:
    adapter = make_adapter()
    sample = adapter.record_to_sample({
        'filename': 'sample-1',
        'prompt_text': 'This is the reference speaker.',
        'WavPath': 'prompt-wavs/sample.wav',
        'text': 'Please synthesize this sentence.',
        'ans': 'wavs/sample.wav',
        'audio': {
            'bytes': b'RIFF....WAVE',
            'path': 'sample.wav',
        },
    })

    assert sample.target == 'Please synthesize this sentence.'
    assert sample.metadata['wer_language'] == 'en'
    assert isinstance(sample.input[0], ChatMessageUser)
    assert isinstance(sample.input[0].content[0], ContentAudio)
    assert sample.input[0].content[0].audio.startswith('data:audio/wav;base64,')
    assert isinstance(sample.input[0].content[1], ContentText)
    assert 'Target text: Please synthesize this sentence.' in sample.input[0].content[1].text


def test_inference_end_saves_audio_output(tmp_path: Any) -> None:
    adapter = make_adapter()
    sample = Sample(
        input=[ChatMessageUser(content='hello')],
        target='hello',
        id=0,
        group_id=0,
        metadata={'filename': 'sample-1'},
    )
    audio = base64.b64encode(b'fake-audio').decode('utf-8')
    output = ModelOutput(
        model='mock',
        choices=[ChatCompletionChoice.from_content([ContentAudio(audio=audio, format='wav')])],
    )

    class MockModel:
        name = 'mock'

    state = adapter._on_inference_end(MockModel(), sample, output, str(tmp_path))

    generated_audio = state.metadata['generated_audio_path']
    assert generated_audio.endswith('sample-1_0.wav')
    assert output.completion == generated_audio
    with open(generated_audio, 'rb') as f:
        assert f.read() == b'fake-audio'


def test_match_score_uses_generated_audio_path(monkeypatch: Any) -> None:
    adapter = make_adapter(metric_list=[{'audio_wer': {'api_key': 'test-key', 'api_base': 'https://asr.test/v1'}}])
    sample = Sample(input=[ChatMessageUser(content='hello')], target='hello', metadata={'wer_language': 'en'})
    state = TaskState(model='mock', sample=sample, completed=True)
    state.metadata['generated_audio_path'] = 'data:audio/wav;base64,' + base64.b64encode(b'audio').decode('utf-8')

    def mock_post(
        self: Any,
        url: str,
        headers: dict[str, str],
        files: dict[str, Any],
        data: dict[str, str],
        timeout: float,
    ) -> Any:

        class Response:

            text = '{"text": "hello"}'

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, str]:
                return {'text': 'hello'}

        assert url == 'https://asr.test/v1/audio/transcriptions'
        assert headers['Authorization'] == 'Bearer test-key'
        assert data['model'] == 'whisper-1'
        return Response()

    monkeypatch.setattr('requests.Session.post', mock_post)
    score = adapter.match_score('', '', 'hello', state)

    assert score.value['audio_wer'] == 0.0
    assert score.metadata['transcription'] == 'hello'


def test_audio_wer_accepts_full_transcription_endpoint(monkeypatch: Any) -> None:

    def mock_post(
        self: Any,
        url: str,
        headers: dict[str, str],
        files: dict[str, Any],
        data: dict[str, str],
        timeout: float,
    ) -> Any:

        class Response:

            text = 'hello'

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, str]:
                return {'text': 'hello'}

        assert url == 'https://asr.test/v1/audio/transcriptions'
        return Response()

    monkeypatch.setattr('requests.Session.post', mock_post)
    metric = AudioWER(api_base='https://asr.test/v1/audio/transcriptions', api_key='test-key')
    score = metric('data:audio/wav;base64,' + base64.b64encode(b'audio').decode('utf-8'), 'hello')

    assert score == 0.0
    assert metric.transcriptions == ['hello']


def test_audio_wer_supports_responses_protocol(monkeypatch: Any) -> None:

    def mock_post(
        self: Any,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> Any:

        class Response:

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return {
                    'output': [{
                        'type': 'message',
                        'content': [{
                            'type': 'output_text',
                            'text': 'hello',
                        }],
                    }]
                }

        assert url == 'https://ark.test/api/v3/responses'
        assert headers['Authorization'] == 'Bearer test-key'
        content = json['input'][0]['content']
        assert content[0]['type'] == 'input_audio'
        assert content[0]['audio_url'].startswith('data:audio/wav;base64,')
        assert content[1]['type'] == 'input_text'
        return Response()

    monkeypatch.setattr('requests.Session.post', mock_post)
    metric = AudioWER(
        api_base='https://ark.test/api/v3',
        api_key='test-key',
        model='doubao-seed-2-0-lite-260428',
        api_protocol='responses',
    )
    score = metric('data:audio/wav;base64,' + base64.b64encode(b'audio').decode('utf-8'), 'hello')

    assert score == 0.0
    assert metric.transcriptions == ['hello']


def test_audio_wer_rejects_unsupported_protocol() -> None:
    with pytest.raises(ValueError, match='Unsupported audio_wer api_protocol'):
        AudioWER(api_base='https://asr.test/v1', api_key='test-key', api_protocol='streaming')
