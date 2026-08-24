from io import BytesIO
from PIL import Image

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage, ContentText
from evalscope.benchmarks.blink.blink_adapter import BLINKAdapter
from evalscope.config import TaskConfig


def _adapter() -> BLINKAdapter:
    return BLINKAdapter(
        benchmark_meta=BenchmarkMeta(name='blink', dataset_id='evalscope/BLINK', eval_split='val'),
        task_config=TaskConfig(datasets=['blink']),
    )


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format=image_format)
    return buffer.getvalue()


def test_blink_normalizes_indexed_images_in_order() -> None:
    sample = _adapter().record_to_sample({
        'prompt': 'Which image is brighter?',
        'choices': ['the first image', 'the second image'],
        'answer': '(B)',
        'image_1': {'bytes': _image_bytes('PNG')},
        'image_2': {'bytes': _image_bytes('JPEG')},
    })

    content = sample.input[0].content
    assert [item.type for item in content] == ['text', 'image', 'image']
    assert isinstance(content[0], ContentText)
    assert content[0].text.endswith('\n\nWhich image is brighter?')
    images = [item for item in content if isinstance(item, ContentImage)]
    assert images[0].image.startswith('data:image/png;base64,')
    assert images[1].image.startswith('data:image/jpeg;base64,')
    assert sample.target == 'B'


def test_blink_supports_plural_media_paths() -> None:
    sample = _adapter().record_to_sample({
        'prompt': 'Compare the images.',
        'choices': ['same', 'different'],
        'answer': 'A',
        'images': [{'path': 'first.jpg'}, {'url': 'https://example.com/second.jpg'}],
    })

    images = [item.image for item in sample.input[0].content if isinstance(item, ContentImage)]
    assert images == ['first.jpg', 'https://example.com/second.jpg']


def test_blink_prefers_indexed_media_to_plural_media() -> None:
    sample = _adapter().record_to_sample({
        'prompt': 'Use the indexed image.',
        'choices': ['yes', 'no'],
        'answer': 'A',
        'image_1': {'path': 'indexed.jpg'},
        'images': [{'path': 'plural.jpg'}],
    })

    assert [item.image for item in sample.input[0].content if isinstance(item, ContentImage)] == ['indexed.jpg']
