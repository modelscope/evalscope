import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from evalscope.api.messages.content import Content, ContentAudio, ContentImage, ContentText, ContentVideo
from evalscope.utils import get_logger
from evalscope.utils.io_utils import bytes_to_base64, compress_image_to_limit, parse_size
from evalscope.utils.media_utils import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    guess_audio_format,
    guess_video_format,
)
from .default_data_adapter import DefaultDataAdapter

logger = get_logger()

MediaType = Literal['audio', 'image', 'video']

# Media types whose payload carries an explicit format hint; images have no ContentImage.format field.
SUPPORTED_MEDIA_FORMATS: Dict[str, Tuple[str, ...]] = {
    'audio': SUPPORTED_AUDIO_FORMATS,
    'video': SUPPORTED_VIDEO_FORMATS,
}


class VisionLanguageAdapter(DefaultDataAdapter):
    """Adapter for vision-language benchmarks. e.g., image captioning, visual question answering, etc."""

    MAX_IMAGES: int = 100
    MAX_VIDEOS: int = 100
    MAX_AUDIOS: int = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Optional image size limit; None means no compression is applied.
        # Can be configured via dataset_args: {'<benchmark_name>': {'max_image_bytes': <int|str>}}
        # Accepts integers (bytes) or human-readable strings like '5mb', '500kb', '1.5gb'.
        self._max_image_bytes: Optional[int] = parse_size(self._benchmark_meta.max_image_bytes)
        self._missing_media_warned: Set[str] = set()

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
            else:
                self._warn_missing_media(media_type, media_num)

            last_end = match.end()

        # Add remaining text after last image
        if last_end < len(text):
            remaining_text = text[last_end:]
            if remaining_text.strip():
                content_list.append(ContentText(text=remaining_text))

        return content_list

    def _warn_missing_media(self, media_type: str, media_num: int) -> None:
        """Report an unresolved placeholder once per index, so sparse datasets stay diagnosable."""
        placeholder = f'<{media_type} {media_num}>'
        if placeholder in self._missing_media_warned:
            return
        self._missing_media_warned.add(placeholder)
        logger.warning(f'No {media_type} supplied for {placeholder}; dropping the placeholder from the prompt.')

    def _normalize_media_value(
        self,
        media_value: Union[str, Dict[str, Any]],
        *,
        media_type: MediaType,
        media_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize one raw media cell into a payload dict whose ``url`` key holds the resolved media.

        Accepts a plain string (local path, URL or data URI), an undecoded Hugging Face dict
        (``{'path': ...}`` or ``{'bytes': ...}``, the latter converted to a base64 data URI) or an
        API-ready dict keyed by ``url`` / ``data`` / ``<media_type>``. Keys other than the payload
        are preserved, so video hints such as ``start`` / ``end`` / ``fps`` survive.

        Args:
            media_value (str | dict): Raw cell value taken from the record.
            media_type (MediaType): Media family the cell belongs to.
            media_format (str, optional): Format hint from the ``<media_type>_<n>_format`` column.

        Returns:
            Dict[str, Any]: Payload with a resolved ``url`` key plus ``format`` and any extra keys.

        Raises:
            ValueError: The dict holds no recognizable payload, the bytes do not decode to
                *media_type*, or the resolved format is unsupported for *media_type*.
        """
        if isinstance(media_value, str):
            normalized_value: Dict[str, Any] = {}
            url = media_value
        else:
            normalized_value = dict(media_value)
            # popping both keys keeps a stale path out of the payload when bytes win
            bytes_obj = normalized_value.pop('bytes', None)
            path = normalized_value.pop('path', None)

            if isinstance(bytes_obj, (bytes, bytearray)):
                if media_type == 'image':
                    url = self._image_bytes_to_base64(bytes_obj, guess_mimetype=True)
                else:
                    url = bytes_to_base64(bytes_obj, add_header=True, guess_mimetype=True)
                # fail closed: a mimetype outside the expected family means the column is mislabeled
                if not url.startswith(f'data:{media_type}/'):
                    raise ValueError(f'{media_type.title()} is invalid as base64 {media_type}, got {url[:30]!r}...')
            elif isinstance(path, str):
                url = path
            else:
                # for future-migration, openai format, include 'url', 'data', or 'video'/'image'/'audio' keys
                payload = media_value.get('url') or media_value.get('data') or media_value.get(media_type)
                if not isinstance(payload, str):
                    raise ValueError(
                        f'Expected media dict with one of "path", "bytes", "url", "data", or "{media_type}" keys, got {media_value!r}'
                    )
                url = payload

        normalized_value.setdefault('format', media_format)
        resolved_format = normalized_value['format']
        supported_formats = SUPPORTED_MEDIA_FORMATS.get(media_type)
        if resolved_format and supported_formats and resolved_format not in supported_formats:
            raise ValueError(
                f'Unsupported {media_type} format {resolved_format!r}, expected one of {supported_formats}'
            )
        normalized_value['url'] = url
        return normalized_value

    def _extract_media(self, record: Dict[str, Any], media_type: MediaType) -> Dict[int, Dict[str, Any]]:
        """Collect and normalize every media cell of *media_type* found in one record.

        Indexed columns (``image_1`` .. ``image_<MAX_IMAGES>``) take precedence over the plural list
        column (``images``), which is intentionally unbounded because long-context benchmarks such as
        OCR may carry more media than the indexed limit allows. Absent cells (``None`` or ``''``) are
        skipped, leaving their placeholders unresolved rather than failing the whole record.

        Args:
            record (dict): Raw record from the dataset.
            media_type (MediaType): Media family to collect.

        Returns:
            Dict[int, Dict[str, Any]]: 1-based media index to payload, shaped by
                :meth:`_normalize_media_value`.

        Raises:
            TypeError: The plural column is not a list, or a cell is neither a string nor a dict.
            ValueError: A cell cannot be normalized; the message names the offending index.
        """
        max_media = {'audio': self.MAX_AUDIOS, 'image': self.MAX_IMAGES, 'video': self.MAX_VIDEOS}[media_type]
        raw_media: Dict[int, Any] = {}
        media_formats: Dict[int, Optional[str]] = {}

        for index in range(1, max_media + 1):
            value = record.get(f'{media_type}_{index}')
            if value is None or value == '':
                continue
            raw_media[index] = value
            if media_type in SUPPORTED_MEDIA_FORMATS:
                media_formats[index] = record.get(f'{media_type}_{index}_format')

        if not raw_media:
            media_list = record.get(f'{media_type}s')
            if not media_list:
                return {}
            if not isinstance(media_list, list):
                raise TypeError(f'"{media_type}s" must be a list of media values, got {type(media_list).__name__}.')
            raw_media = {i + 1: media for i, media in enumerate(media_list) if media is not None and media != ''}

        media_map: Dict[int, Dict[str, Any]] = {}
        for index, value in raw_media.items():
            # Hugging Face media columns surface undecoded values as dicts.
            if not isinstance(value, (str, dict)):
                raise TypeError(
                    f'Expect {index}th {media_type} as string (path, URL, or base64) or undecoded dict, '
                    f'got {type(value)}'
                )

            try:
                media_map[index] = self._normalize_media_value(
                    value,
                    media_type=media_type,
                    media_format=media_formats.get(index),
                )
            except ValueError as e:
                raise ValueError(f'Failed to parse {index}th {media_type}: {value!r}') from e
        return media_map

    @staticmethod
    def _content_video_from_value(video_value: Union[str, Dict[str, Any]]) -> ContentVideo:
        if isinstance(video_value, dict):
            video = video_value.get('url')
            if not video:
                raise ValueError(f'Video payload must include "url", got {video_value!r}')
            video_format = video_value.get('format') or guess_video_format(video)
            start = video_value.get('start')
            end = video_value.get('end')
            fps = video_value.get('fps')
        else:
            video = video_value
            video_format = guess_video_format(video)
            start = None
            end = None
            fps = None
        return ContentVideo(video=video, format=video_format, start=start, end=end, fps=fps)

    @staticmethod
    def _content_audio_from_value(audio_value: Union[str, Dict[str, Any]]) -> ContentAudio:
        if isinstance(audio_value, dict):
            audio = audio_value.get('url')
            if not audio:
                raise ValueError(f'Audio payload must include "url", got {audio_value!r}')
            audio_format = audio_value.get('format') or guess_audio_format(audio)
        else:
            audio = audio_value
            audio_format = guess_audio_format(audio)
        return ContentAudio(audio=audio, format=audio_format)

    @staticmethod
    def _content_image_from_value(image_value: Union[str, Dict[str, Any]]) -> ContentImage:
        if isinstance(image_value, dict):
            image = image_value.get('url')
            if not image:
                raise ValueError(f'Image payload must include "url", got {image_value!r}')
        else:
            image = image_value
        return ContentImage(image=image)
