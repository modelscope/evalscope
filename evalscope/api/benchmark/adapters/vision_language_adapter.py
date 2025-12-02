import re
from typing import Dict, List

from evalscope.api.messages.content import Content, ContentImage, ContentText
from .default_data_adapter import DefaultDataAdapter


class VisionLanguageAdapter(DefaultDataAdapter):
    """Adapter for vision-language benchmarks. e.g., image captioning, visual question answering, etc."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _parse_text_with_images(self, text: str, image_map: Dict[int, str]) -> List[Content]:
        """
        Parse text and replace <image x> placeholders with actual images.

        Args:
            text (str): Text containing <image x> placeholders
            image_map (dict): Mapping from image number to base64 encoded image

        Returns:
            list: List of Content objects (text and images interleaved)
        """
        content_list: List[Content] = []

        # Pattern to match <image x> where x is a number
        pattern = r'<image[_ ](\d+)>'
        last_end = 0

        for match in re.finditer(pattern, text):
            # Add text before the image placeholder
            if match.start() > last_end:
                text_segment = text[last_end:match.start()]
                if text_segment.strip():
                    content_list.append(ContentText(text=text_segment))

            # Add the image
            image_num = int(match.group(1))
            if image_num in image_map:
                content_list.append(ContentImage(image=image_map[image_num]))

            last_end = match.end()

        # Add remaining text after last image
        if last_end < len(text):
            remaining_text = text[last_end:]
            if remaining_text.strip():
                content_list.append(ContentText(text=remaining_text))

        return content_list
