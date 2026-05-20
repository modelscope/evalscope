import re
from typing import Any, Dict, List, Optional, Union

from evalscope.api.messages.content import Content, ContentImage, ContentText, ContentVideo
from evalscope.utils.io_utils import bytes_to_base64, compress_image_to_limit, parse_size
from evalscope.utils.url_utils import guess_video_format
from .default_data_adapter import DefaultDataAdapter


class VisionLanguageAdapter(DefaultDataAdapter):
    """Adapter for vision-language benchmarks. e.g., image captioning, visual question answering, etc."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Optional image size limit; None means no compression is applied.
        # Can be configured via dataset_args: {'<benchmark_name>': {'max_image_bytes': <int|str>}}
        # Accepts integers (bytes) or human-readable strings like '5mb', '500kb', '1.5gb'.
        self._max_image_bytes: Optional[int] = parse_size(self._benchmark_meta.max_image_bytes)

    def _image_bytes_to_base64(self, image_bytes: bytes, default_format: str = 'png') -> str:
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
            return bytes_to_base64(compressed_bytes, format=fmt, add_header=True)
        return bytes_to_base64(image_bytes, format=default_format, add_header=True)

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
        image_map: Optional[Dict[int, str]] = None,
        video_map: Optional[Dict[int, Union[str, Dict[str, Any]]]] = None,
    ) -> List[Content]:
        """
        Parse text and replace <image x>/<video x> placeholders with media content.
        """
        image_map = image_map or {}
        video_map = video_map or {}
        content_list: List[Content] = []

        pattern = r'<(image|video)[_ ](\d+)>'
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
                content_list.append(ContentImage(image=image_map[media_num]))
            elif media_type == 'video' and video_map.get(media_num):
                content_list.append(self._content_video_from_value(video_map[media_num]))

            last_end = match.end()

        # Add remaining text after last image
        if last_end < len(text):
            remaining_text = text[last_end:]
            if remaining_text.strip():
                content_list.append(ContentText(text=remaining_text))

        return content_list

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
