"""URL and data-URI predicates and conversion helpers."""

import base64
import mimetypes
import re
from typing import Literal, Optional

from evalscope.utils.download_utils import download_url

VideoFormat = Literal['mp4', 'mpeg', 'mov', 'avi']
SUPPORTED_VIDEO_FORMATS: tuple[VideoFormat, ...] = ('mp4', 'mpeg', 'mov', 'avi')
VIDEO_FORMAT_TO_MIME_TYPE: dict[VideoFormat, str] = {
    'mp4': 'video/mp4',
    'mpeg': 'video/mpeg',
    'mov': 'video/quicktime',
    'avi': 'video/x-msvideo',
}


def is_http_url(url: str) -> bool:
    return url.startswith('http://') or url.startswith('https://')


def is_data_uri(url: str) -> bool:
    pattern = r'^data:([^;]+);base64,.*'
    return re.match(pattern, url) is not None


def data_uri_mime_type(data_url: str) -> Optional[str]:
    match = re.match(r'^data:([^;]+);.*', data_url)
    return match.group(1) if match else None


def data_uri_to_base64(data_uri: str) -> str:
    return re.sub(r'^data:[^,]+,', '', data_uri)


def file_as_data(file: str, default_mime_type: str = 'image/png') -> tuple[bytes, str]:
    """Resolve a data URI, HTTP URL, or local path to raw bytes and a MIME type."""
    if is_data_uri(file):
        mime_type = data_uri_mime_type(file) or default_mime_type
        return base64.b64decode(data_uri_to_base64(file)), mime_type

    # Guess mime type; need strict=False for webp images.
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


def video_format_to_mime_type(video_format: VideoFormat) -> str:
    """Return the MIME type for a supported video format."""
    from evalscope.utils.media_utils import video_format_to_mime_type as impl

    return impl(video_format)


def guess_video_format(video: Optional[str], default: VideoFormat = 'mp4') -> VideoFormat:
    """Infer a supported video format from a data URI, URL, or local path."""
    from evalscope.utils.media_utils import guess_video_format as impl

    return impl(video, default)


def video_as_data_uri(video: str, video_format: Optional[VideoFormat] = None) -> str:
    """Convert a video URL or path to a data URI."""
    from evalscope.utils.media_utils import video_as_data_uri as impl

    return impl(video, video_format)


__all__ = [
    'SUPPORTED_VIDEO_FORMATS',
    'VIDEO_FORMAT_TO_MIME_TYPE',
    'VideoFormat',
    'data_uri_mime_type',
    'data_uri_to_base64',
    'download_url',
    'file_as_data',
    'file_as_data_uri',
    'guess_video_format',
    'is_data_uri',
    'is_http_url',
    'video_as_data_uri',
    'video_format_to_mime_type',
]
