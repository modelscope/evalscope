"""Video format detection and media MIME type helpers."""

import mimetypes
import os
from typing import Literal, Optional, cast

from evalscope.utils.uri_utils import data_uri_mime_type, file_as_data_uri, is_data_uri

VideoFormat = Literal['mp4', 'mpeg', 'mov', 'avi']
SUPPORTED_VIDEO_FORMATS: tuple[VideoFormat, ...] = ('mp4', 'mpeg', 'mov', 'avi')
VIDEO_FORMAT_TO_MIME_TYPE: dict[VideoFormat, str] = {
    'mp4': 'video/mp4',
    'mpeg': 'video/mpeg',
    'mov': 'video/quicktime',
    'avi': 'video/x-msvideo',
}


def video_format_to_mime_type(video_format: VideoFormat) -> str:
    return VIDEO_FORMAT_TO_MIME_TYPE[video_format]


def guess_video_format(video: Optional[str], default: VideoFormat = 'mp4') -> VideoFormat:
    """Infer a supported video format from a data URI, URL, or local path."""
    if not video:
        return default

    mime_type = data_uri_mime_type(video)
    if not mime_type:
        path_like = video.split('?', 1)[0].split('#', 1)[0]
        mime_type, _ = mimetypes.guess_type(path_like, strict=False)

    if mime_type and mime_type.startswith('video/'):
        subtype = mime_type.split('/', 1)[1].lower()
        if subtype == 'quicktime':
            return 'mov'
        if subtype == 'x-msvideo':
            return 'avi'
        if subtype in SUPPORTED_VIDEO_FORMATS:
            return cast(VideoFormat, subtype)

    ext = os.path.splitext(video.split('?', 1)[0].split('#', 1)[0])[1].lstrip('.').lower()
    if ext in SUPPORTED_VIDEO_FORMATS:
        return cast(VideoFormat, ext)

    return default


def video_as_data_uri(video: str, video_format: Optional[VideoFormat] = None) -> str:
    if is_data_uri(video):
        return video
    video_format = video_format or guess_video_format(video)
    return file_as_data_uri(video, default_mime_type=video_format_to_mime_type(video_format))


__all__ = [
    'SUPPORTED_VIDEO_FORMATS',
    'VIDEO_FORMAT_TO_MIME_TYPE',
    'VideoFormat',
    'guess_video_format',
    'video_as_data_uri',
    'video_format_to_mime_type',
]
