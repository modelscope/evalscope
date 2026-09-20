import re
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional

from PIL import Image

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.io_utils import PIL_to_base64


@register_dataset('mmmu_multi_image')
class MMMUMultiImageDatasetPlugin(DatasetPluginBase):
    """Build real multi-image stress-test requests from the MMMU validation set."""

    dataset_id = 'AI-ModelScope/MMMU'
    subsets = (
        'Accounting',
        'Agriculture',
        'Architecture_and_Engineering',
        'Art',
        'Art_Theory',
        'Basic_Medical_Science',
        'Biology',
        'Chemistry',
        'Clinical_Medicine',
        'Computer_Science',
        'Design',
        'Diagnostics_and_Laboratory_Medicine',
        'Economics',
        'Electronics',
        'Energy_and_Power',
        'Finance',
        'Geography',
        'History',
        'Literature',
        'Manage',
        'Marketing',
        'Materials',
        'Math',
        'Mechanical_Engineering',
        'Music',
        'Pharmacy',
        'Physics',
        'Psychology',
        'Public_Health',
        'Sociology',
    )
    max_images = 7
    min_images = 2

    # Modes the shared JPEG encoder (the PIL_to_base64 default) can write.
    # Real MMMU rows also carry incompatible modes such as RGBA, so they are
    # converted to RGB before encoding.
    JPEG_COMPATIBLE_MODES = frozenset({'L', 'RGB', 'CMYK', 'YCbCr'})
    IMAGE_PLACEHOLDER_PATTERN = re.compile(r'<image[_ ](\d+)>')

    def __init__(self, query_parameters: Arguments) -> None:
        if query_parameters.tokenize_prompt:
            raise ValueError(
                '--tokenize-prompt is not supported with the mmmu_multi_image dataset. '
                'The dataset produces multimodal messages that cannot be represented as a flat token-ID list.'
            )
        super().__init__(query_parameters)

    @staticmethod
    def _to_pil_image(value: Any) -> Optional[Image.Image]:
        """Normalize common dataset image representations to a PIL image."""
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            raw_bytes = value.get('bytes')
            if raw_bytes:
                with Image.open(BytesIO(raw_bytes)) as image:
                    return image.copy()
            path = value.get('path')
            if path:
                with Image.open(path) as image:
                    return image.copy()
        return None

    @staticmethod
    def _to_jpeg_compatible(image: Image.Image) -> Image.Image:
        """Convert non-JPEG-compatible image modes to RGB before encoding.

        ``PIL_to_base64`` defaults to JPEG, which refuses modes such as
        ``RGBA``/``LA``/``P``. Converting those images to ``RGB`` keeps the
        existing JPEG data-URL encoding path usable for MMMU samples.
        """
        if image.mode in MMMUMultiImageDatasetPlugin.JPEG_COMPATIBLE_MODES:
            return image
        return image.convert('RGB')

    def _collect_image_urls(self, item: Dict[str, Any]) -> Dict[int, str]:
        """Collect non-empty ``image_1`` ... ``image_7`` fields by source index."""
        image_urls: Dict[int, str] = {}
        for index in range(1, self.max_images + 1):
            image = self._to_pil_image(item.get(f'image_{index}'))
            if image is None:
                continue
            image_urls[index] = PIL_to_base64(self._to_jpeg_compatible(image), add_header=True)
        return image_urls

    @staticmethod
    def _build_prompt(item: Dict[str, Any]) -> str:
        """Keep MMMU question text and options without changing benchmark semantics."""
        prompt = str(item['question'])
        options = item.get('options')
        if options and options != '[]':
            prompt = f'{prompt}\nOptions: {options}'
        return prompt

    def _create_message_with_placeholders(self, prompt: str, image_urls: Dict[int, str]) -> Dict[str, Any]:
        """Replace MMMU image placeholders with ordered OpenAI image content blocks."""
        matches = list(self.IMAGE_PLACEHOLDER_PATTERN.finditer(prompt))
        if not matches:
            return self.create_message(text=prompt, image_urls=list(image_urls.values()))

        content: List[Dict[str, Any]] = []
        last_end = 0
        for match in matches:
            text_segment = prompt[last_end : match.start()]
            if text_segment.strip():
                content.append({'type': 'text', 'text': text_segment})
            image_url = image_urls.get(int(match.group(1)))
            if image_url:
                content.append({'type': 'image_url', 'image_url': {'url': image_url}})
            last_end = match.end()

        remaining_text = prompt[last_end:]
        if remaining_text.strip():
            content.append({'type': 'text', 'text': remaining_text})
        if not content:
            return self.create_message(text=prompt, image_urls=list(image_urls.values()))
        return {'role': 'user', 'content': content}

    def build_messages(self) -> Iterator[List[Dict]]:
        """Yield eligible samples from every MMMU subject in round-robin order."""
        dataset_iterators = [
            iter(
                self.load_hub_dataset(
                    dataset_id=self.dataset_id,
                    split='validation',
                    subset=subset,
                )
            )
            for subset in self.subsets
        ]

        while dataset_iterators:
            active_iterators = []
            for dataset_iterator in dataset_iterators:
                try:
                    item = next(dataset_iterator)
                except StopIteration:
                    continue

                active_iterators.append(dataset_iterator)
                image_urls = self._collect_image_urls(item)
                if len(image_urls) < self.min_images:
                    continue

                message = self._create_message_with_placeholders(self._build_prompt(item), image_urls)
                yield [message]

            dataset_iterators = active_iterators
