from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage, ContentText
from evalscope.benchmarks.blink.blink_adapter import BLINKAdapter
from evalscope.config import TaskConfig


def _adapter() -> BLINKAdapter:
    return BLINKAdapter(
        benchmark_meta=BenchmarkMeta(name='blink', dataset_id='evalscope/BLINK', eval_split='val'),
        task_config=TaskConfig(datasets=['blink']),
    )


def test_blink_accepts_undecoded_image_bytes() -> None:
    sample = _adapter().record_to_sample({
        'prompt': 'Which image is brighter?',
        'choices': ['the first image', 'the second image'],
        'answer': '(B)',
        'image_1': {'bytes': b'not-an-image'},
    })

    content = sample.input[0].content
    assert isinstance(content[0], ContentText)
    assert isinstance(content[1], ContentImage)
    assert content[1].image.startswith('data:image/jpeg;base64,')
    assert sample.target == 'B'


def test_blink_accepts_plural_media_paths() -> None:
    sample = _adapter().record_to_sample({
        'prompt': 'Compare the images.',
        'choices': ['same', 'different'],
        'answer': 'A',
        'images': [{'path': 'first.jpg'}, {'url': 'https://example.com/second.jpg'}],
    })

    content = sample.input[0].content
    image_content = [item for item in content if isinstance(item, ContentImage)]
    assert [item.image for item in image_content] == ['first.jpg', 'https://example.com/second.jpg']
    assert sample.target == 'A'
