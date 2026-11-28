import argparse
import base64
import io
import json
import math
import os
import struct
import sys
import threading
import types
import wave
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default as default_email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from openai import APIStatusError, BadRequestError, RateLimitError

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.model import GenerateConfig, get_model, get_model_with_task_config
from evalscope.arguments import add_argument
from evalscope.benchmarks.librispeech.librispeech_adapter import LibriSpeechAdapter
from evalscope.config import DEFAULT_API_EVAL_BATCH_SIZE, TaskConfig
from evalscope.constants import EvalType
from evalscope.models.speech2text_model import Speech2TextAPI


def _make_wav_bytes(sample_rate: int = 16000, duration_s: float = 0.25, freq_hz: float = 440.0) -> bytes:
    buf = io.BytesIO()
    num_samples = int(sample_rate * duration_s)
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            val = int(16000 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
            frames.extend(struct.pack('<h', val))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    raw_msg = f'Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n'.encode('utf-8') + body
    msg = BytesParser(policy=default_email_policy).parsebytes(raw_msg)
    fields: dict[str, str] = {}
    files: dict[str, dict[str, Any]] = {}
    for part in msg.iter_parts():
        name = part.get_param('name', header='content-disposition')
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b''
        if filename is not None:
            files[name] = {
                'filename': filename,
                'content_type': part.get_content_type(),
                'bytes': payload,
            }
        else:
            fields[name] = payload.decode('utf-8')
    return fields, files


class _TranscriptionServiceHandler(BaseHTTPRequestHandler):

    requests: list[dict[str, Any]] = []
    response_kind = 'json'
    response_text = 'hello world'
    transient_failures = 0

    def _send_json(self, status: int, response: dict[str, Any]) -> None:
        body = json.dumps(response).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('x-request-id', 'stt-request-id')
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('x-request-id', 'stt-request-id')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(content_length)
        content_type = self.headers.get('Content-Type', '')
        fields, files = _parse_multipart(content_type, body)
        parsed_url = urlsplit(self.path)
        self.requests.append({
            'path': parsed_url.path,
            'query': parse_qs(parsed_url.query),
            'headers': {k.lower(): v for k, v in self.headers.items()},
            'fields': fields,
            'files': files,
        })

        if self.response_kind == 'error':
            self._send_json(503, {'error': {'message': 'transcription backend unavailable', 'type': 'server_error'}})
            return
        if self.response_kind == 'bad_request':
            self._send_json(400, {'error': {'message': 'unsupported audio format', 'type': 'invalid_request_error'}})
            return
        if len(self.requests) <= self.transient_failures:
            self._send_json(429, {'error': {'message': 'rate limited', 'type': 'rate_limit_error'}})
            return

        if self.response_kind == 'text':
            self._send_text(200, self.response_text)
        elif self.response_kind == 'verbose_json':
            self._send_json(
                200,
                {
                    'task': 'transcribe',
                    'language': 'english',
                    'duration': 0.25,
                    'text': self.response_text,
                },
            )
        elif self.response_kind == 'empty_text':
            self._send_json(200, {'text': ''})
        else:
            self._send_json(200, {'text': self.response_text})

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def _transcription_service(
    response_kind: str = 'json',
    response_text: str = 'hello world',
    transient_failures: int = 0,
) -> Iterator[str]:
    _TranscriptionServiceHandler.requests = []
    _TranscriptionServiceHandler.response_kind = response_kind
    _TranscriptionServiceHandler.response_text = response_text
    _TranscriptionServiceHandler.transient_failures = transient_failures
    server = ThreadingHTTPServer(('127.0.0.1', 0), _TranscriptionServiceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f'http://{host}:{port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _audio_message(wav_bytes: bytes, fmt: str = 'wav') -> list[ChatMessageUser]:
    mime = 'audio/mpeg' if fmt == 'mp3' else f'audio/{fmt}'
    data_uri = f'data:{mime};base64,' + base64.b64encode(wav_bytes).decode('utf-8')
    return [
        ChatMessageUser(
            content=[
                ContentText(text='Please recognize the speech and only output the recognized content:'),
                ContentAudio(audio=data_uri, format=fmt),
            ]
        )
    ]


def test_remote_speech2text_uses_openai_transcriptions_endpoint(caplog: pytest.LogCaptureFixture) -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(response_kind='verbose_json', response_text='hello fourier') as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1/audio/transcriptions',
            api_key='test-key',
            config=GenerateConfig(
                temperature=0.0,
                extra_headers={'X-Gateway-Token': 'gw-123'},
                extra_query={'api-version': '2026-01'},
                extra_body={'diarize': False},
                vendor_flag='fast',
            ),
            model_args={
                'language': 'en',
                'prompt': 'Oruk Fourier',
                'response_format': 'verbose_json',
                'precision': 'bfloat16',
                'max_retries': 3,
            },
            memoize=False,
        )
        output = model.generate(_audio_message(wav_bytes))

    request = _TranscriptionServiceHandler.requests[0]
    assert request['path'] == '/v1/audio/transcriptions'
    assert request['query'] == {'api-version': ['2026-01']}
    assert request['headers']['authorization'] == 'Bearer test-key'
    assert request['headers']['x-gateway-token'] == 'gw-123'
    assert request['fields']['model'] == 'oruk-fourier-2b'
    assert request['fields']['language'] == 'en'
    assert request['fields']['prompt'] == 'Oruk Fourier'
    assert request['fields']['response_format'] == 'verbose_json'
    assert float(request['fields']['temperature']) == 0.0
    assert request['fields']['vendor_flag'] == 'fast'
    assert request['fields']['diarize'] == 'false'
    assert request['files']['file']['filename'] == 'audio.wav'
    assert request['files']['file']['content_type'] == 'audio/wav'
    assert request['files']['file']['bytes'] == wav_bytes
    assert output.completion == 'hello fourier'
    assert output.metadata['request_id'] == 'stt-request-id'
    assert output.metadata['duration'] == 0.25
    assert output.message.perf_metrics is not None
    assert output.message.perf_metrics.latency >= 0.0
    assert 'precision' in caplog.text
    assert 'max_retries' in caplog.text


def test_remote_speech2text_normalizes_chat_completions_url() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(response_text='normalized url') as service_url:
        model = get_model(
            model='whisper-1',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1/chat/completions',
            api_key='test-key',
            memoize=False,
        )
        output = model.generate(_audio_message(wav_bytes))

    assert _TranscriptionServiceHandler.requests[0]['path'] == '/v1/audio/transcriptions'
    assert output.completion == 'normalized url'


def test_remote_speech2text_accepts_plain_text_and_preserves_empty_transcript() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(response_kind='text', response_text='plain transcript') as service_url:
        model = get_model(
            model='whisper-1',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            model_args={'response_format': 'text'},
            memoize=False,
        )
        output = model.generate(_audio_message(wav_bytes))
    assert output.completion == 'plain transcript'

    with _transcription_service(response_kind='empty_text') as service_url:
        model = get_model(
            model='whisper-1',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        empty_output = model.generate(_audio_message(wav_bytes))
    assert empty_output.completion == ''


def test_remote_speech2text_handles_local_file_path_and_mp3_format(tmp_path: Path) -> None:
    audio_file = tmp_path / 'sample.mp3'
    audio_file.write_bytes(b'ID3-fake-mp3-bytes')
    with _transcription_service(response_text='from local file') as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        output = model.generate([ChatMessageUser(content=[ContentAudio(audio=str(audio_file), format='mp3')])])

    uploaded = _TranscriptionServiceHandler.requests[0]['files']['file']
    assert uploaded['filename'] == 'sample.mp3'
    assert uploaded['content_type'] == 'audio/mpeg'
    assert uploaded['bytes'] == b'ID3-fake-mp3-bytes'
    assert output.completion == 'from local file'


def test_remote_speech2text_prefers_evalscope_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALSCOPE_API_KEY', 'evalscope-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    wav_bytes = _make_wav_bytes()

    with _transcription_service() as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='EMPTY',
            memoize=False,
        )
        model.generate(_audio_message(wav_bytes))

    assert _TranscriptionServiceHandler.requests[0]['headers']['authorization'] == 'Bearer evalscope-key'


def test_remote_speech2text_empty_key_falls_back_to_empty_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('EVALSCOPE_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    wav_bytes = _make_wav_bytes()

    with _transcription_service() as service_url:
        model = get_model(
            model='whisper-1',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='EMPTY',
            memoize=False,
        )
        model.generate(_audio_message(wav_bytes))

    assert _TranscriptionServiceHandler.requests[0]['headers']['authorization'] == 'Bearer EMPTY'


def test_remote_speech2text_retries_transient_errors() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(transient_failures=1, response_text='recovered') as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=2, retry_interval=0),
            memoize=False,
        )
        output = model.generate(_audio_message(wav_bytes))

    assert len(_TranscriptionServiceHandler.requests) == 2
    assert output.completion == 'recovered'


def test_remote_speech2text_stops_after_configured_retries() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(transient_failures=3) as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=2, retry_interval=0),
            memoize=False,
        )
        with pytest.raises(RateLimitError):
            model.generate(_audio_message(wav_bytes))

    assert len(_TranscriptionServiceHandler.requests) == 2


def test_remote_speech2text_does_not_retry_client_errors() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(response_kind='bad_request') as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=3, retry_interval=0),
            memoize=False,
        )
        with pytest.raises(BadRequestError, match='unsupported audio format'):
            model.generate(_audio_message(wav_bytes))

    assert len(_TranscriptionServiceHandler.requests) == 1


def test_remote_speech2text_preserves_server_error_body() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(response_kind='error') as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=1, retry_interval=0),
            memoize=False,
        )
        with pytest.raises(APIStatusError, match='transcription backend unavailable'):
            model.generate(_audio_message(wav_bytes))


def test_remote_speech2text_rejects_missing_or_empty_audio() -> None:
    with _transcription_service() as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        with pytest.raises(ValueError, match='ContentAudio'):
            model.generate([ChatMessageUser(content='no audio here')])

        with pytest.raises(ValueError, match='empty'):
            model.generate([ChatMessageUser(content=[ContentAudio(audio='data:audio/wav;base64,', format='wav')])])


def test_task_config_and_cli_support_speech2text() -> None:
    service_cfg = TaskConfig(
        model='oruk-fourier-2b',
        eval_type=EvalType.SPEECH2TEXT,
        api_url='https://speech-api.oruk.ai/v1',
        datasets=['librispeech'],
    )
    assert service_cfg.eval_type == EvalType.SPEECH2TEXT
    assert service_cfg.eval_batch_size == DEFAULT_API_EVAL_BATCH_SIZE

    local_cfg = TaskConfig(
        model='oruk/oruk-fourier-2b',
        eval_type=EvalType.SPEECH2TEXT,
        datasets=['librispeech'],
    )
    assert local_cfg.eval_type == EvalType.SPEECH2TEXT
    assert local_cfg.eval_batch_size == 1

    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args(['--model', 'oruk/oruk-fourier-2b', '--eval-type', 'speech2text', '--datasets', 'librispeech'])
    assert args.eval_type == EvalType.SPEECH2TEXT


def test_librispeech_adapter_end_to_end_with_speech2text_service() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(response_text='The quick brown fox.') as service_url:
        task_cfg = TaskConfig(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            api_url=f'{service_url}/v1',
            api_key='test-key',
            datasets=['librispeech'],
        )
        model = get_model_with_task_config(task_cfg)
        adapter = LibriSpeechAdapter(
            benchmark_meta=BenchmarkMeta(
                name='librispeech',
                dataset_id='lmms-lab/Librispeech-concat',
                prompt_template='Please recognize the speech and only output the recognized content:',
            ),
            task_config=task_cfg,
        )
        sample = adapter.record_to_sample({
            'audio': {'bytes': wav_bytes},
            'transcript': 'the quick brown fox',
            'audio_id': 'test-001',
            'audio_duration': 0.25,
        })
        output = model.generate(sample.input)
        score = adapter.match_score(output.completion, output.completion, sample.target, None)

    assert score.value == {'wer': 0.0}


def test_local_speech2text_checks_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_import(*args: Any, **kwargs: Any) -> bool:
        raise ImportError('missing local deps')

    monkeypatch.setattr('evalscope.models.speech2text_model.check_import', _fail_import)
    with pytest.raises(ImportError, match='missing local deps'):
        get_model(
            model='oruk/oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            memoize=False,
        )


def test_local_speech2text_qwen3_asr_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeTensor:
        def __init__(self, data: np.ndarray):
            self._data = data
            self.shape = data.shape

        def is_floating_point(self) -> bool:
            return np.issubdtype(self._data.dtype, np.floating)

        def to(self, device: Any = None, dtype: Any = None) -> '_FakeTensor':
            captured.setdefault('moved', []).append((device, dtype))
            return self

        def __getitem__(self, item: Any) -> Any:
            return self._data[item]

    class _FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: Any) -> None:
            return None

    fake_torch = types.ModuleType('torch')
    fake_torch.bfloat16 = 'bf16'
    fake_torch.float16 = 'fp16'
    fake_torch.float32 = 'fp32'
    fake_torch.inference_mode = lambda: _FakeNoGrad()

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Any:
            captured['config_call'] = (model_id, kwargs)
            return types.SimpleNamespace(model_type='qwen3_asr', architectures=['Qwen3ASRForConditionalGeneration'])

    class _FakeProcessor:
        feature_extractor = types.SimpleNamespace(sampling_rate=16000)

        def apply_transcription_request(self, **kwargs: Any) -> dict[str, Any]:
            captured['processor_request'] = kwargs
            return {
                'input_ids': _FakeTensor(np.array([[10, 11, 12]], dtype=np.int64)),
                'input_features': _FakeTensor(np.zeros((1, 80, 10), dtype=np.float32)),
            }

        def decode(self, token_ids: Any, return_format: str = 'raw', **kwargs: Any) -> str:
            captured['decode'] = (token_ids.tolist(), return_format)
            return 'transcribed from qwen3 asr'

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeProcessor:
            captured['processor_call'] = (model_id, kwargs)
            return _FakeProcessor()

    class _FakeQwen3ASRModel:
        device = 'cpu'
        dtype = 'bf16'

        @classmethod
        def from_pretrained(cls, model_id: str, dtype: Any = None, **kwargs: Any) -> '_FakeQwen3ASRModel':
            captured['model_call'] = (model_id, dtype, kwargs)
            return cls()

        def to(self, device: Any) -> '_FakeQwen3ASRModel':
            captured['model_to'] = device
            return self

        def eval(self) -> '_FakeQwen3ASRModel':
            captured['eval_called'] = True
            return self

        def generate(self, **kwargs: Any) -> np.ndarray:
            captured['generate_kwargs'] = {k: v for k, v in kwargs.items() if k not in {'input_ids', 'input_features'}}
            return np.array([[10, 11, 12, 201, 202]], dtype=np.int64)

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoConfig = _FakeAutoConfig
    fake_transformers.AutoProcessor = _FakeAutoProcessor
    fake_transformers.Qwen3ASRForConditionalGeneration = _FakeQwen3ASRModel

    monkeypatch.setattr('evalscope.models.speech2text_model.check_import', lambda *a, **k: True)
    monkeypatch.setattr('evalscope.models.speech2text_model.get_device', lambda: 'cpu')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)

    wav_bytes = _make_wav_bytes(sample_rate=24000, duration_s=0.1)
    model = get_model(
        model='oruk/oruk-fourier-2b',
        eval_type=EvalType.SPEECH2TEXT,
        config=GenerateConfig(max_tokens=128),
        model_args={
            'revision': '8c2a8871f1304aa2f634da4fb95e24438c4945b9',
            'precision': 'bfloat16',
            'language': 'en',
        },
        memoize=False,
    )
    output = model.generate(_audio_message(wav_bytes))

    assert captured['config_call'] == (
        'oruk/oruk-fourier-2b',
        {'trust_remote_code': False, 'revision': '8c2a8871f1304aa2f634da4fb95e24438c4945b9'},
    )
    assert captured['model_call'][0] == 'oruk/oruk-fourier-2b'
    assert captured['model_call'][1] == 'bf16'
    assert captured['processor_request']['language'] == 'en'
    assert len(captured['processor_request']['audio']) == 1600
    assert captured['generate_kwargs'] == {'max_new_tokens': 128}
    assert captured['decode'] == ([201, 202], 'transcription_only')
    assert captured['eval_called'] is True
    assert output.completion == 'transcribed from qwen3 asr'
    assert output.metadata['backend'] == 'qwen3_asr'


def test_local_speech2text_qwen3_asr_requires_modern_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.ModuleType('torch')
    fake_torch.float32 = 'fp32'

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(model_type='qwen3_asr', architectures=['Qwen3ASRForConditionalGeneration'])

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoConfig = _FakeAutoConfig

    monkeypatch.setattr('evalscope.models.speech2text_model.check_import', lambda *a, **k: True)
    monkeypatch.setattr('evalscope.models.speech2text_model.get_device', lambda: 'cpu')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)

    with pytest.raises(ImportError, match='transformers>=5.21'):
        get_model(
            model='oruk/oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            memoize=False,
        )


def test_local_speech2text_generic_pipeline_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    fake_torch = types.ModuleType('torch')
    fake_torch.float16 = 'fp16'

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> Any:
            return types.SimpleNamespace(model_type='whisper', architectures=['WhisperForConditionalGeneration'])

    def _fake_pipeline(task: str, **kwargs: Any) -> Any:
        captured['pipeline_init'] = (task, kwargs)

        def _run(audio_input: dict[str, Any], **call_kwargs: Any) -> dict[str, str]:
            captured['pipeline_call'] = (audio_input, call_kwargs)
            return {'text': 'whisper transcript'}

        return _run

    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoConfig = _FakeAutoConfig
    fake_transformers.pipeline = _fake_pipeline

    monkeypatch.setattr('evalscope.models.speech2text_model.check_import', lambda *a, **k: True)
    monkeypatch.setattr('evalscope.models.speech2text_model.get_device', lambda: 'cpu')
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)

    wav_bytes = _make_wav_bytes(sample_rate=16000, duration_s=0.1)
    model = get_model(
        model='openai/whisper-tiny',
        eval_type=EvalType.SPEECH2TEXT,
        model_args={'language': 'en', 'chunk_length_s': 30, 'precision': 'float16'},
        memoize=False,
    )
    output = model.generate(_audio_message(wav_bytes))

    assert captured['pipeline_init'][0] == 'automatic-speech-recognition'
    assert captured['pipeline_init'][1]['model'] == 'openai/whisper-tiny'
    assert captured['pipeline_call'][0]['sampling_rate'] == 16000
    assert captured['pipeline_call'][1] == {'chunk_length_s': 30, 'generate_kwargs': {'language': 'en'}}
    assert output.completion == 'whisper transcript'
    assert output.metadata['backend'] == 'pipeline'


def test_strip_asr_text_prefix_fallback() -> None:
    assert Speech2TextAPI._strip_asr_text_prefix('language English<asr_text>hello world') == 'hello world'
    assert Speech2TextAPI._strip_asr_text_prefix('plain text') == 'plain text'


def test_remote_speech2text_zero_retries_still_sends_request() -> None:
    wav_bytes = _make_wav_bytes()
    with _transcription_service(response_text='single attempt') as service_url:
        model = get_model(
            model='oruk-fourier-2b',
            eval_type=EvalType.SPEECH2TEXT,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=0, retry_interval=0),
            memoize=False,
        )
        output = model.generate(_audio_message(wav_bytes))

    assert len(_TranscriptionServiceHandler.requests) == 1
    assert output.completion == 'single attempt'


def test_local_speech2text_ffmpeg_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import soundfile as sf

    def _fail_sf_read(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('unsupported container')

    expected = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)

    class _FakeCompleted:
        stdout = expected.tobytes()

    monkeypatch.setattr(sf, 'read', _fail_sf_read)
    monkeypatch.setattr('subprocess.run', lambda *a, **k: _FakeCompleted())

    waveform, sr = Speech2TextAPI._decode_audio_waveform(b'fake-opus-bytes', target_sr=16000, audio_ext='opus')
    assert sr == 16000
    assert np.allclose(waveform, expected)
