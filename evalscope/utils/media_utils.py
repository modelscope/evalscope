"""Video format detection and media MIME type helpers."""

import mimetypes
import os
from typing import Literal, Optional, cast

from evalscope.utils.uri_utils import data_uri_mime_type, file_as_data_uri, is_data_uri

VideoFormat = Literal['mp4', 'mpeg', 'mov', 'avi']
AudioFormat = Literal['mp3', 'wav']
SUPPORTED_VIDEO_FORMATS: tuple[VideoFormat, ...] = ('mp4', 'mpeg', 'mov', 'avi')
SUPPORTED_AUDIO_FORMATS: tuple[AudioFormat, ...] = ('mp3', 'wav')
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


def guess_audio_format(audio: Optional[str], default: AudioFormat = 'mp3') -> AudioFormat:
    """Infer a supported audio format from a data URI, URL, or local path."""
    if not audio:
        return default

    format_aliases: dict[str, AudioFormat] = {
        'mpeg': 'mp3',
        'mpga': 'mp3',
        'x-mp3': 'mp3',
        'x-wav': 'wav',
        'wave': 'wav',
        'vnd.wave': 'wav',
    }

    mime_type = data_uri_mime_type(audio)
    if not mime_type:
        path_like = audio.split('?', 1)[0].split('#', 1)[0]
        mime_type, _ = mimetypes.guess_type(path_like, strict=False)

    if mime_type and mime_type.startswith('audio/'):
        subtype = mime_type.split('/', 1)[1].lower()
        if subtype in SUPPORTED_AUDIO_FORMATS:
            return cast(AudioFormat, subtype)
        if subtype in format_aliases:
            return format_aliases[subtype]

    ext = os.path.splitext(audio.split('?', 1)[0].split('#', 1)[0])[1].lstrip('.').lower()
    if ext in SUPPORTED_AUDIO_FORMATS:
        return cast(AudioFormat, ext)
    if ext in format_aliases:
        return format_aliases[ext]

    return default


__all__ = [
    'SUPPORTED_AUDIO_FORMATS',
    'SUPPORTED_VIDEO_FORMATS',
    'VIDEO_FORMAT_TO_MIME_TYPE',
    'AudioFormat',
    'VideoFormat',
    'guess_audio_format',
    'guess_video_format',
    'video_as_data_uri',
    'video_format_to_mime_type',
]
