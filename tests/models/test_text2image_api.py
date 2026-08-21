import base64
import json
import pytest
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from evalscope.api.messages import ContentImage
from evalscope.api.model import GenerateConfig, get_model, get_model_with_task_config
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, ModelTask


class _ImageServiceHandler(BaseHTTPRequestHandler):

    requests: list[dict[str, Any]] = []
    image_requests: list[str] = []
    image_bytes = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
    )
    response_kind = 'base64'

    def do_POST(self) -> None:
        content_length = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(content_length))
        self.requests.append({
            'path': self.path,
            'headers': dict(self.headers),
            'payload': payload,
        })
        if self.response_kind == 'base64':
            response = {
                'created': 123,
                'data': [{
                    'b64_json': base64.b64encode(self.image_bytes).decode('utf-8'),
                    'revised_prompt': 'A revised prompt',
                }],
                'usage': {
                    'input_tokens': 4,
                    'output_tokens': 8,
                    'total_tokens': 12,
                },
            }
        elif self.response_kind == 'url':
            host, port = self.server.server_address
            response = {'data': [{'url': f'http://{host}:{port}/generated.png'}]}
        else:
            response = {'data': []}
        body = json.dumps(response).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.image_requests.append(self.path)
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(self.image_bytes)))
        self.end_headers()
        self.wfile.write(self.image_bytes)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def _image_service(response_kind: str = 'base64') -> Iterator[str]:
    _ImageServiceHandler.requests = []
    _ImageServiceHandler.image_requests = []
    _ImageServiceHandler.response_kind = response_kind
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


def test_remote_text2image_uses_openai_compatible_endpoint() -> None:
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
            ),
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
    }
    assert request['headers']['Authorization'] == 'Bearer test-key'

    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes
    assert output.metadata == {
        'created': 123,
        'revised_prompt': 'A revised prompt',
        'usage': {
            'input_tokens': 4,
            'output_tokens': 8,
            'total_tokens': 12,
        },
    }


def test_remote_text2image_downloads_url_response() -> None:
    with _image_service(response_kind='url') as service_url:
        model = get_model(
            model='Qwen-Image',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            api_key=None,
            memoize=False,
        )
        output = model.generate('Draw a lighthouse in a storm.')

    request = _ImageServiceHandler.requests[0]
    assert request['path'] == '/v1/images/generations'
    assert 'Authorization' not in request['headers']
    assert _ImageServiceHandler.image_requests == ['/generated.png']
    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes


def test_remote_text2image_rejects_empty_data() -> None:
    with _image_service(response_kind='empty') as service_url:
        model = get_model(
            model='Qwen-Image',
            eval_type=EvalType.TEXT2IMAGE,
            base_url=f'{service_url}/v1',
            memoize=False,
        )
        with pytest.raises(ValueError, match='non-empty data list'):
            model.generate('Draw an empty response.')


def test_image_generation_service_is_selected_from_task_config() -> None:
    with _image_service() as service_url:
        task_config = TaskConfig(
            model='Qwen-Image',
            model_task=ModelTask.IMAGE_GENERATION,
            api_url=f'{service_url}/v1',
        )
        model = get_model_with_task_config(task_config)
        output = model.generate('Draw a mountain reflected in a lake.')

    assert task_config.eval_type == EvalType.TEXT2IMAGE
    assert _ImageServiceHandler.requests[0]['path'] == '/v1/images/generations'
    content = output.message.content[0]
    assert isinstance(content, ContentImage)
    assert base64.b64decode(content.image) == _ImageServiceHandler.image_bytes


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
