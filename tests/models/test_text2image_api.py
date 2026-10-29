import base64
import glob
import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
from openai import APIStatusError, BadRequestError, RateLimitError

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.messages import ContentImage
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelOutput, get_model, get_model_with_task_config
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, FileConstants, ModelTask
from evalscope.utils.uri_utils import data_uri_to_base64


class _ImageServiceHandler(BaseHTTPRequestHandler):

    requests: list[dict[str, Any]] = []
    image_requests: list[str] = []
    image_bytes = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
    )
    response_kind = 'base64'
    transient_failures = 0

    def _send_json(self, status: int, response: dict[str, Any]) -> None:
        body = json.dumps(response).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('x-request-id', 'request-id')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(content_length))
        self.requests.append({
            'path': self.path,
            'headers': dict(self.headers),
            'payload': payload,
        })

        if self.response_kind == 'error':
            self._send_json(503, {'error': {'message': 'image backend unavailable', 'type': 'server_error'}})
            return
        if self.response_kind == 'bad_request':
            self._send_json(400, {'error': {'message': 'unsupported size', 'type': 'invalid_request_error'}})
            return
        if len(self.requests) <= self.transient_failures:
            self._send_json(429, {'error': {'message': 'rate limited', 'type': 'rate_limit_error'}})
            return

        if self.response_kind in {'base64', 'data_uri', 'invalid_base64', 'non_image_base64'}:
            image_base64 = base64.b64encode(self.image_bytes).decode('utf-8')
            if self.response_kind == 'data_uri':
                image_base64 = f'data:image/png;base64,{image_base64}'
            elif self.response_kind == 'invalid_base64':
                image_base64 = 'not-valid-base64'
            elif self.response_kind == 'non_image_base64':
                image_base64 = base64.b64encode(b'definitely not an image').decode('utf-8')
            response = {
                'created': 123,
                'data': [{
                    'b64_json': image_base64,
                    'revised_prompt': 'A revised prompt',
                }],
                'usage': {
                    'input_tokens': 4,
                    'output_tokens': 8,
                    'total_tokens': 12,
                },
            }
        elif self.response_kind in {'url', 'html_url'}:
            host, port = self.server.server_address
            file_name = 'generated.png' if self.response_kind == 'url' else 'login.html'
            response = {'created': 123, 'data': [{'url': f'http://{host}:{port}/{file_name}'}]}
        elif self.response_kind == 'invalid_url':
            response = {'created': 123, 'data': [{'url': '/tmp/generated.png'}]}
        else:
            response = {'created': 123, 'data': []}
        self._send_json(200, response)

    def do_GET(self) -> None:
        self.image_requests.append(self.path)
        if self.path.endswith('.html'):
            body, content_type = b'<!doctype html><html><body>Sign in</body></html>', 'text/html'
        else:
            body, content_type = self.image_bytes, 'image/png'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def _image_service(response_kind: str = 'base64', transient_failures: int = 0) -> Iterator[str]:
    _ImageServiceHandler.requests = []
    _ImageServiceHandler.image_requests = []
    _ImageServiceHandler.response_kind = response_kind
    _ImageServiceHandler.transient_failures = transient_failures
    server = ThreadingHTTPServer(('127.0.0.1', 0), _ImageServiceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f'http://{host}:{port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _image_bytes(content: Any) -> bytes:
    assert isinstance(content, ContentImage)
    assert content.image.startswith('data:image/png;base64,')
    return base64.b64decode(data_uri_to_base64(content.image))


class _StubModel:
    name = 'test-image-model'


def _t2i_adapter() -> Text2ImageAdapter:
    return Text2ImageAdapter(
        benchmark_meta=BenchmarkMeta(name='general_t2i', dataset_id='general_t2i'),
        task_config=TaskConfig(
            model='test-image-model',
            model_task=ModelTask.IMAGE_GENERATION,
            eval_type=EvalType.TEXT2IMAGE,
            datasets=['general_t2i'],
        ),
    )


def _image_output() -> ModelOutput:
    image_uri = 'data:image/png;base64,' + base64.b64encode(_ImageServiceHandler.image_bytes).decode('utf-8')
    return ModelOutput(
        model='test-image-model',
        choices=[ChatCompletionChoice.from_content(content=[ContentImage(image=image_uri)])],
    )


def test_remote_text2image_uses_openai_compatible_endpoint(caplog: pytest.LogCaptureFixture) -> None:
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1/images/generations',
            api_key='test-key',
            config=GenerateConfig(
                width=1024,
                height=1024,
                n=1,
                quality='low',
                vendor_option='fast',
            ),
            model_args={
                'pipeline_cls': 'FluxPipeline',
                'use_safetensors': True,
                'variant': 'fp16',
            },
            memoize=False,
        )
        output = model.generate('Draw a red panda reading a book.')

    request = _ImageServiceHandler.requests[0]
    assert request['path'] == '/v1/images/generations'
    assert request['payload'] == {
        'model': 'test-image-model',
        'prompt': 'Draw a red panda reading a book.',
        'n': 1,
        'quality': 'low',
        'size': '1024x1024',
        'vendor_option': 'fast',
    }
    assert request['headers']['Authorization'] == 'Bearer test-key'
    assert 'pipeline_cls' not in request['payload']
    assert 'use_safetensors' not in request['payload']
    assert 'variant' not in request['payload']
    assert 'Ignoring model_args unsupported by the OpenAI client' in caplog.text
    assert 'pipeline_cls, use_safetensors, variant' in caplog.text

    assert _image_bytes(output.message.content[0]) == _ImageServiceHandler.image_bytes
    assert output.metadata == {
        'created': 123,
        'request_id': 'request-id',
        'revised_prompt': 'A revised prompt',
        'usage': {
            'input_tokens': 4,
            'output_tokens': 8,
            'total_tokens': 12,
        },
    }


def test_remote_text2image_downloads_url_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('EVALSCOPE_API_KEY', raising=False)
    with _image_service(response_kind='url') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='EMPTY',
            memoize=False,
        )
        output = model.generate('Draw a lighthouse in a storm.')

    request = _ImageServiceHandler.requests[0]
    assert request['path'] == '/v1/images/generations'
    assert request['headers']['Authorization'] == 'Bearer EMPTY'
    assert _ImageServiceHandler.image_requests == ['/generated.png']
    assert _image_bytes(output.message.content[0]) == _ImageServiceHandler.image_bytes


def test_remote_text2image_empty_key_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    monkeypatch.delenv('EVALSCOPE_API_KEY', raising=False)
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='EMPTY',
            memoize=False,
        )
        model.generate('Draw a cabin in the snow.')

    assert _ImageServiceHandler.requests[0]['headers']['Authorization'] == 'Bearer openai-key'


def test_remote_text2image_prefers_evalscope_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    monkeypatch.setenv('EVALSCOPE_API_KEY', 'evalscope-key')
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='EMPTY',
            memoize=False,
        )
        model.generate('Draw a cabin in the snow.')

    assert _ImageServiceHandler.requests[0]['headers']['Authorization'] == 'Bearer evalscope-key'


def test_remote_text2image_normalizes_chat_completions_url() -> None:
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1/chat/completions',
            api_key='test-key',
            memoize=False,
        )
        model.generate('Draw a paper boat.')

    assert _ImageServiceHandler.requests[0]['path'] == '/v1/images/generations'


def test_remote_text2image_retries_transient_errors(caplog: pytest.LogCaptureFixture) -> None:
    with _image_service(transient_failures=2) as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=3, retry_interval=0),
            model_args={'max_retries': 3},
            memoize=False,
        )
        output = model.generate('Draw a sunrise.')

    # SDK-internal retries are disabled, so each generation_config.retries attempt is exactly one request.
    assert len(_ImageServiceHandler.requests) == 3
    assert 'use generation_config.retries instead' in caplog.text
    assert _image_bytes(output.message.content[0]) == _ImageServiceHandler.image_bytes


def test_remote_text2image_stops_after_configured_retries() -> None:
    with _image_service(transient_failures=10) as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=3, retry_interval=0),
            memoize=False,
        )
        with pytest.raises(RateLimitError):
            model.generate('Draw a crowded queue.')

    assert len(_ImageServiceHandler.requests) == 3


def test_remote_text2image_does_not_retry_client_errors() -> None:
    with _image_service(response_kind='bad_request') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=3, retry_interval=0),
            memoize=False,
        )
        with pytest.raises(BadRequestError) as exc_info:
            model.generate('Draw an unsupported size.')

    assert exc_info.value.body == {'message': 'unsupported size', 'type': 'invalid_request_error'}
    assert len(_ImageServiceHandler.requests) == 1


def test_remote_text2image_zero_retries_still_sends_request() -> None:
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=0, retry_interval=0),
            memoize=False,
        )
        output = model.generate('Draw a single request.')

    assert len(_ImageServiceHandler.requests) == 1
    assert isinstance(output.message.content[0], ContentImage)


def test_remote_text2image_preserves_error_body() -> None:
    with _image_service(response_kind='error') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(retries=1, retry_interval=0),
            memoize=False,
        )
        with pytest.raises(APIStatusError) as exc_info:
            model.generate('Draw an unavailable image.')

    assert exc_info.value.body == {'message': 'image backend unavailable', 'type': 'server_error'}
    assert len(_ImageServiceHandler.requests) == 1


@pytest.mark.parametrize('response_kind', ['base64', 'data_uri'])
def test_remote_text2image_accepts_base64_variants(response_kind: str) -> None:
    with _image_service(response_kind=response_kind) as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        output = model.generate('Draw a blue circle.')

    assert _image_bytes(output.message.content[0]) == _ImageServiceHandler.image_bytes


def test_remote_text2image_rejects_invalid_base64() -> None:
    with _image_service(response_kind='invalid_base64') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        with pytest.raises(ValueError, match='invalid base64 image data'):
            model.generate('Draw invalid image data.')


def test_remote_text2image_rejects_base64_non_image_bytes() -> None:
    with _image_service(response_kind='non_image_base64') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        with pytest.raises(ValueError, match=r'not a valid image \(from b64_json\)'):
            model.generate('Draw non-image bytes.')


def test_remote_text2image_rejects_url_returning_html() -> None:
    with _image_service(response_kind='html_url') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        with pytest.raises(ValueError, match=r'not a valid image \(from http://.*/login\.html\)'):
            model.generate('Draw behind a login page.')

    assert _ImageServiceHandler.image_requests == ['/login.html']


def test_remote_text2image_forwards_diffusion_params_in_body() -> None:
    with _image_service() as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            config=GenerateConfig(num_inference_steps=8, guidance_scale=3.5),
            memoize=False,
        )
        model.generate('Draw a quick sketch.')

    payload = _ImageServiceHandler.requests[0]['payload']
    assert payload['num_inference_steps'] == 8
    assert payload['guidance_scale'] == 3.5


def test_remote_text2image_rejects_non_http_url() -> None:
    with _image_service(response_kind='invalid_url') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key='test-key',
            memoize=False,
        )
        with pytest.raises(ValueError, match='non-HTTP image URL'):
            model.generate('Draw a local file path.')


def test_remote_text2image_rejects_empty_data() -> None:
    with _image_service(response_kind='empty') as service_url:
        model = get_model(
            model='test-image-model',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            memoize=False,
        )
        with pytest.raises(ValueError, match='non-empty data list'):
            model.generate('Draw an empty response.')


def test_image_generation_service_is_selected_from_task_config() -> None:
    with _image_service() as service_url:
        task_config = TaskConfig(
            model='test-image-model',
            model_task=ModelTask.IMAGE_GENERATION,
            api_url=f'{service_url}/v1',
        )
        model = get_model_with_task_config(task_config)
        output = model.generate('Draw a mountain reflected in a lake.')

    assert task_config.eval_type == EvalType.TEXT2IMAGE
    assert _ImageServiceHandler.requests[0]['path'] == '/v1/images/generations'
    # The service default config keeps the image size but not the diffusion-only knobs.
    assert _ImageServiceHandler.requests[0]['payload'] == {
        'model': 'test-image-model',
        'prompt': 'Draw a mountain reflected in a lake.',
        'size': '1024x1024',
    }
    assert _image_bytes(output.message.content[0]) == _ImageServiceHandler.image_bytes


def test_image_generation_service_corrects_openai_api_eval_type() -> None:
    with _image_service() as service_url:
        task_config = TaskConfig(
            model='test-image-model',
            model_task=ModelTask.IMAGE_GENERATION,
            # The pairing reported in issue #1040: an Images endpoint with eval_type=openai_api,
            # which would otherwise post to /chat/completions.
            api_url=f'{service_url}/v1/images/generations',
            eval_type=EvalType.OPENAI_API,
            api_key='EMPTY',
        )
        model = get_model_with_task_config(task_config)
        output = model.generate('Draw a lighthouse at dusk.')

    assert task_config.eval_type == EvalType.TEXT2IMAGE
    assert _ImageServiceHandler.requests[0]['path'] == '/v1/images/generations'
    assert _image_bytes(output.message.content[0]) == _ImageServiceHandler.image_bytes


def test_text_generation_service_keeps_openai_api_eval_type() -> None:
    task_config = TaskConfig(
        model='test-chat-model',
        api_url='http://127.0.0.1:1/v1',
        eval_type=EvalType.OPENAI_API,
        api_key='EMPTY',
    )

    assert task_config.eval_type == EvalType.OPENAI_API


def test_image_editing_service_keeps_its_eval_type() -> None:
    task_config = TaskConfig(
        model='test-edit-model',
        model_task=ModelTask.IMAGE_GENERATION,
        api_url='http://127.0.0.1:1/v1',
        eval_type=EvalType.IMAGE_EDITING,
        api_key='EMPTY',
    )

    assert task_config.eval_type == EvalType.IMAGE_EDITING


def test_text2image_adapter_saves_data_uri_output(tmp_path: Path) -> None:
    adapter = _t2i_adapter()
    output = _image_output()
    sample = Sample(input='Draw a red dot.', id=1, group_id=0)

    adapter._on_inference_end(_StubModel(), sample, output, str(tmp_path))

    with open(sample.metadata[FileConstants.IMAGE_PATH], 'rb') as f:
        assert f.read() == _ImageServiceHandler.image_bytes


def test_text2image_adapter_saves_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _t2i_adapter()
    worker_count = 4
    barrier = threading.Barrier(worker_count)
    real_makedirs = os.makedirs

    def synchronized_makedirs(name: str, *args: Any, **kwargs: Any) -> None:
        # Release every worker into the real call at once, so the shared image directory
        # is always created by several workers simultaneously rather than by chance.
        barrier.wait(timeout=10)
        real_makedirs(name, *args, **kwargs)

    monkeypatch.setattr(
        'evalscope.api.benchmark.adapters.text2image_adapter.os.makedirs',
        synchronized_makedirs,
    )
    samples = [Sample(input=f'Draw dot {index}.', id=index, group_id=index) for index in range(worker_count)]
    failures: list[BaseException] = []

    def save(sample: Sample) -> None:
        try:
            adapter._on_inference_end(_StubModel(), sample, _image_output(), str(tmp_path))
        except BaseException as ex:  # noqa: BLE001 - recorded so the assertion can report it
            failures.append(ex)

    threads = [threading.Thread(target=save, args=(sample, )) for sample in samples]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not failures, failures
    saved = sorted(glob.glob(os.path.join(str(tmp_path), 'images', '*.png')))
    assert len(saved) == worker_count


def test_local_text2image_still_checks_aigc_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    checked_modules: list[str] = []

    def reject_local_dependencies(
        module_name: list[str],
        package: str,
        raise_error: bool,
        feature_name: str,
    ) -> None:
        checked_modules.extend(module_name)
        raise ImportError('local dependencies are unavailable')

    monkeypatch.setattr('evalscope.models.text2image_model.check_import', reject_local_dependencies)

    with pytest.raises(ImportError, match='local dependencies are unavailable'):
        get_model(
            model='Qwen-Image',
            eval_type=EvalType.TEXT2IMAGE,
            memoize=False,
        )

    assert checked_modules == ['torch', 'torchvision', 'diffusers']
