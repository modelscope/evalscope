"""HTTP URL, data URI and file conversion helpers."""

import base64
import mimetypes
import re
from typing import Optional
from urllib.parse import urlsplit


def is_http_url(url: str) -> bool:
    return url.startswith('http://') or url.startswith('https://')


def sanitize_endpoint(url: Optional[str]) -> Optional[str]:
    if not isinstance(url, str) or not url.strip():
        return None
    candidate = url.strip()
    try:
        parsed = urlsplit(candidate if '://' in candidate else f'//{candidate}')
        host, port = parsed.hostname, parsed.port
    except ValueError:
        return None
    if not host or any(char.isspace() for char in host):
        return None
    host = host.lower()
    if ':' in host:
        host = f'[{host}]'
    return f'{host}:{port}' if port is not None else host


def validate_http_endpoint(url: str) -> None:
    if not isinstance(url, str) or not url.strip():
        raise ValueError('API URL is required')
    invalid_url_message = 'API URL must be a valid HTTP or HTTPS URL with a valid host and port'
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        raise ValueError(invalid_url_message) from None
    if parsed.scheme not in ('http', 'https') or sanitize_endpoint(url) is None:
        raise ValueError(invalid_url_message)
    if any(char.isspace() for char in url.strip()):
        raise ValueError(invalid_url_message)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError('Put authentication in API Key, not in the URL')


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
    'sanitize_endpoint',
    'validate_http_endpoint',
]
