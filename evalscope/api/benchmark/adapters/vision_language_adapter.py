import re
from typing import Any, Dict, List, Literal, Optional, Union

from evalscope.api.messages.content import Content, ContentAudio, ContentImage, ContentText, ContentVideo
from evalscope.utils.io_utils import bytes_to_base64, compress_image_to_limit, parse_size
from evalscope.utils.media_utils import guess_audio_format, guess_video_format
from .default_data_adapter import DefaultDataAdapter


class VisionLanguageAdapter(DefaultDataAdapter):
    """Adapter for vision-language benchmarks. e.g., image captioning, visual question answering, etc."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Optional image size limit; None means no compression is applied.
        # Can be configured via dataset_args: {'<benchmark_name>': {'max_image_bytes': <int|str>}}
        # Accepts integers (bytes) or human-readable strings like '5mb', '500kb', '1.5gb'.
        self._max_image_bytes: Optional[int] = parse_size(self._benchmark_meta.max_image_bytes)

    def _image_bytes_to_base64(
        self, image_bytes: bytes, default_format: str = 'png', guess_mimetype: bool = False
    ) -> str:
        """Convert raw image bytes to a base64 data-URI, compressing first if needed.

        This is the recommended helper for subclasses that obtain images as raw
        bytes.  It applies the optional size-limit compression configured via
        ``max_image_bytes`` before base64-encoding.

        Args:
            image_bytes (bytes): Raw image bytes.
            default_format (str): Image format used when no compression is
                applied.  Defaults to ``'png'``.

        Returns:
            str: Base64-encoded data-URI string with MIME header.
        """
        if self._max_image_bytes is not None:
            compressed_bytes, fmt = compress_image_to_limit(image_bytes, self._max_image_bytes)
            # compress_image_to_limit returns fmt='png' when no compression was applied,
            # which is a sentinel value — not the actual image format.  In that case,
            # fall back to the caller's default_format for the correct MIME type.
            if fmt == 'png':
                fmt = default_format
            return bytes_to_base64(compressed_bytes, format=fmt, add_header=True, guess_mimetype=guess_mimetype)
        return bytes_to_base64(image_bytes, format=default_format, add_header=True, guess_mimetype=guess_mimetype)

    def _parse_text_with_images(self, text: str, image_map: Dict[int, str]) -> List[Content]:
        """
        Parse text and replace <image x> placeholders with actual images.

        Args:
            text (str): Text containing <image x> placeholders
            image_map (dict): Mapping from image number to base64 encoded image

        Returns:
            list: List of Content objects (text and images interleaved)
        """
        return self._parse_text_with_media(text=text, image_map=image_map)

    def _parse_text_with_media(
        self,
        text: str,
        image_map: Optional[Dict[int, Union[str, Dict[str, Any]]]] = None,
        video_map: Optional[Dict[int, Union[str, Dict[str, Any]]]] = None,
        audio_map: Optional[Dict[int, Union[str, Dict[str, Any]]]] = None,
    ) -> List[Content]:
        """
        Parse text and replace <image x>/<video x>/<audio x> placeholders with media content.
        """
        image_map = image_map or {}
        video_map = video_map or {}
        audio_map = audio_map or {}
        content_list: List[Content] = []

        pattern = r'<(image|video|audio)[_ ](\d+)>'
        last_end = 0

        for match in re.finditer(pattern, text):
            # Add text before the image placeholder
            if match.start() > last_end:
                text_segment = text[last_end:match.start()]
                if text_segment.strip():
                    content_list.append(ContentText(text=text_segment))

            media_type = match.group(1)
            media_num = int(match.group(2))
            if media_type == 'image' and image_map.get(media_num):
                content_list.append(self._content_image_from_value(image_map[media_num]))
            elif media_type == 'video' and video_map.get(media_num):
                content_list.append(self._content_video_from_value(video_map[media_num]))
            elif media_type == 'audio' and audio_map.get(media_num):
                content_list.append(self._content_audio_from_value(audio_map[media_num]))

            last_end = match.end()

        # Add remaining text after last image
        if last_end < len(text):
            remaining_text = text[last_end:]
            if remaining_text.strip():
                content_list.append(ContentText(text=remaining_text))

        return content_list

    def _normalize_media_value(
        self,
        media_value: Union[str, Dict[str, Any]],
        *,
        media_type: Literal['audio', 'image', 'video'],
        media_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(media_value, str):
            return {'url': media_value, 'format': media_format}

        normalized_value = dict(media_value)

        # prefer bytes, which skips another read
        if isinstance((bytes_obj := media_value.get('bytes')), (bytes, bytearray)):
            if media_type == 'image':
                base64_url = self._image_bytes_to_base64(bytes_obj, guess_mimetype=True)
            else:
                base64_url = bytes_to_base64(bytes_obj, add_header=True, guess_mimetype=True)

            if not base64_url.startswith(f'data:{media_type}/'):
                raise ValueError(f'{media_type.title()} is invalid as base64 {media_type}, got {base64_url[:30]!r}...')
            normalized_value.pop('bytes')
            normalized_value.setdefault('format', media_format)
            return normalized_value | {'url': base64_url}

        if isinstance((path := media_value.get('path')), str):
            normalized_value.pop('path')
            normalized_value.setdefault('format', media_format)
            return normalized_value | {'url': path}

        # for future-migration, openai format, include 'url', 'data', or 'video'/'image'/'audio' keys
        fallback = (media_value.get('url') or media_value.get('data') or media_value.get(media_type))
        if isinstance(fallback, str):
            normalized_value.setdefault('format', media_format)
            return normalized_value | {'url': fallback}

        raise ValueError('Expected undecoded dict of {"path": ""} or {"bytes": b"..."}'
                         f', got {media_value!r}')

    def _extract_media(self, record: Dict[str, Any], mtype: Literal['audio', 'image',
                                                                    'video']) -> Dict[int, Union[str, Dict[str, Any]]]:
        medias: dict[int, Any] = {}
        mformats: dict[int, Optional[str]] = {}

        # prefer 'image_n' > 'images', 'video_n' > 'videos', 'audio_n' > 'audios'
        # where n is 1..MAX_IMAGES/MAX_VIDEOS/MAX_AUDIOS
        max_n_media: int = getattr(self, f'MAX_{mtype}S'.upper(), 100)
        if any(record.get(f'{mtype}_{i + 1}') for i in range(max_n_media)):
            for i in range(max_n_media):
                medias[i + 1] = record.get(f'{mtype}_{i + 1}')
                mformats[i + 1] = record.get(f'{mtype}_{i + 1}_format')
        elif media_list := record.get(f'{mtype}s'):
            if not isinstance(media_list, list):
                raise TypeError(f'"{mtype}s" must be a list of media values, got {type(media_list).__name__}.')
            # intentionally allow unlimited list of media, when 'xxx_n' is not feasible/readable.
            # e.g. long-context benchmarks like OCR may have >100 images.
            for i, media in enumerate(media_list):
                medias[i + 1] = media
        else:
            return {}

        # normalize raw media values into a consistent payload shape.
        media_map: Dict[int, Union[str, Dict[str, Any]]] = {}
        for k, v in medias.items():
            if v is None:
                continue

            # Hugging Face Media columns may surface undecoded values as dicts.
            # https://huggingface.co/docs/datasets/about_dataset_features#image-feature
            # https://huggingface.co/docs/datasets/about_dataset_features#audio-feature
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Video
            if not isinstance(v, (str, dict)):
                raise TypeError(
                    f'Expect {k}th {mtype} as string (path, URL, or base64) or undecoded dict, got {type(v)}'
                )

            try:
                media_map[k] = self._normalize_media_value(
                    v,
                    media_type=mtype,
                    media_format=mformats.get(k),
                )
            except ValueError as e:
                raise ValueError(f'Failed to parse {k}th {mtype}: {v!r}') from e
        return media_map

    @staticmethod
    def _content_video_from_value(video_value: Union[str, Dict[str, Any]]) -> ContentVideo:
        if isinstance(video_value, dict):
            video = video_value.get('url') or video_value.get('video') or video_value.get('data')
            if not video:
                raise ValueError('Video field must include one of "url", "video", or "data".')
            video_format = video_value.get('format') or guess_video_format(str(video))
            start = video_value.get('start')
            end = video_value.get('end')
            fps = video_value.get('fps')
        else:
            video = video_value
            video_format = guess_video_format(video)
            start = None
            end = None
            fps = None
        return ContentVideo(video=str(video), format=video_format, start=start, end=end, fps=fps)

    @staticmethod
    def _content_audio_from_value(audio_value: Union[str, Dict[str, Any]]) -> ContentAudio:
        if isinstance(audio_value, dict):
            audio = audio_value.get('url') or audio_value.get('audio') or audio_value.get('data')
            if not audio:
                raise ValueError('Audio field must include one of "url", "audio", or "data".')
            audio_format = audio_value.get('format') or guess_audio_format(str(audio))
        else:
            audio = audio_value
            audio_format = guess_audio_format(audio)
        # pydantic may complain if audio format is invalid
        return ContentAudio(audio=str(audio), format=audio_format)

    @staticmethod
    def _content_image_from_value(image_value: Union[str, Dict[str, Any]]) -> ContentImage:
        if isinstance(image_value, dict):
            image = image_value.get('url') or image_value.get('image') or image_value.get('data')
            if not image:
                raise ValueError('Image field must include one of "url", "image", or "data".')
        else:
            image = image_value
        # pydantic may complain if image format is invalid
        return ContentImage(image=str(image))
