"""HTTP URL, data URI and file conversion helpers."""

import base64
import mimetypes
import re
from typing import Optional


def is_http_url(url: str) -> bool:
    return url.startswith('http://') or url.startswith('https://')


def is_data_uri(url: str) -> bool:
    return re.match(r'^data:[^;]+;base64,', url) is not None


def data_uri_mime_type(data_url: str) -> Optional[str]:
    match = re.match(r'^data:([^;]+);.*', data_url)
    return match.group(1) if match else None


def data_uri_to_base64(data_uri: str) -> str:
    return re.sub(r'^data:[^,]+,', '', data_uri)


def file_as_data(file: str, default_mime_type: str = 'image/png') -> tuple[bytes, str]:
    """Resolve a data URI, HTTP URL or local path to raw bytes and a MIME type."""
    if is_data_uri(file):
        mime_type = data_uri_mime_type(file) or default_mime_type
        return base64.b64decode(data_uri_to_base64(file)), mime_type

    guessed_type, _ = mimetypes.guess_type(file, strict=False)
    mime_type = guessed_type or default_mime_type

    if is_http_url(file):
        import requests

        response = requests.get(file, timeout=30)
        response.raise_for_status()
        return response.content, mime_type

    with open(file, 'rb') as f:
        return f.read(), mime_type


def file_as_data_uri(file: str, default_mime_type: str = 'image/png') -> str:
    if is_data_uri(file):
        return file
    file_bytes, mime_type = file_as_data(file, default_mime_type=default_mime_type)
    base64_file = base64.b64encode(file_bytes).decode('utf-8')
    return f'data:{mime_type};base64,{base64_file}'


__all__ = [
    'data_uri_mime_type',
    'data_uri_to_base64',
    'file_as_data',
    'file_as_data_uri',
    'is_data_uri',
    'is_http_url',
]
