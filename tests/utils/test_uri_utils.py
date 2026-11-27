import base64
from pathlib import Path

import pytest
import requests

from evalscope.utils.uri_utils import file_as_data, file_as_data_uri, is_http_url


@pytest.mark.parametrize('scheme', ['http', 'https', 'HTTP', 'HTTPS', 'hTtP', 'hTtPs'])
def test_http_media_preserves_original_url(scheme: str, monkeypatch: pytest.MonkeyPatch) -> None:
    url = f'{scheme}://example.test/Images/Photo.PNG?key=MiXeD'
    payload = b'image bytes'
    calls: list[tuple[str, int]] = []

    def get(source: str, *, timeout: int) -> requests.Response:
        calls.append((source, timeout))
        response = requests.Response()
        response.status_code = 200
        response._content = payload
        return response

    monkeypatch.setattr(requests, 'get', get)

    assert file_as_data(url) == (payload, 'image/png')
    encoded = base64.b64encode(payload).decode('ascii')
    assert file_as_data_uri(url) == f'data:image/png;base64,{encoded}'
    assert calls == [(url, 30), (url, 30)]


@pytest.mark.parametrize('source', ['image.png', '/tmp/image.png', 'file:///tmp/image.png', 'ftp://example.test/image.png'])
def test_non_http_sources_are_not_http_urls(source: str) -> None:
    assert not is_http_url(source)


def test_local_file_and_data_uri_do_not_make_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b'local image bytes'
    path = tmp_path / 'image.png'
    path.write_bytes(payload)
    encoded = base64.b64encode(payload).decode('ascii')
    data_uri = f'data:image/png;base64,{encoded}'

    def unexpected_request(*args: object, **kwargs: object) -> None:
        pytest.fail('Local files and data URIs must not make HTTP requests')

    monkeypatch.setattr(requests, 'get', unexpected_request)

    assert file_as_data(str(path)) == (payload, 'image/png')
    assert file_as_data(data_uri) == (payload, 'image/png')
    assert file_as_data_uri(str(path)) == data_uri
    assert file_as_data_uri(data_uri) == data_uri
