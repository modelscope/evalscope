import pytest
from io import BytesIO
from PIL import Image

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage, ContentText
from evalscope.benchmarks.olympiad_bench import olympiad_bench_adapter
from evalscope.config import TaskConfig


def _adapter(monkeypatch: pytest.MonkeyPatch) -> olympiad_bench_adapter.OlympiadBenchAdapter:
    monkeypatch.setattr(olympiad_bench_adapter, 'check_import', lambda *args, **kwargs: None)
    return olympiad_bench_adapter.OlympiadBenchAdapter(
        benchmark_meta=BenchmarkMeta(name='olympiad_bench', dataset_id='AI-ModelScope/OlympiadBench', eval_split='train'),
        task_config=TaskConfig(datasets=['olympiad_bench']),
    )


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format=image_format)
    return buffer.getvalue()


def test_olympiad_bench_normalizes_indexed_images(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _adapter(monkeypatch).record_to_sample({
        'id': 'sample',
        'question': 'Use <image_1> and <image_2>.',
        'language': 'English',
        'subject': 'Math',
        'question_type': 'open',
        'answer_type': 'Numerical',
        'final_answer': ['1'],
        'image_1': {'bytes': None, 'path': 'diagram.png'},
        'image_2': {'bytes': _image_bytes('JPEG')},
        'image_9': {'path': 'last-supported.png'},
        'image_10': {'path': 'unsupported.png'},
    })

    content = sample.input[0].content
    assert [item.type for item in content] == ['text', 'image', 'image', 'image']
    assert isinstance(content[0], ContentText)
    assert '[image_1]' in content[0].text
    assert '[image_2]' in content[0].text
    images = [item for item in content if isinstance(item, ContentImage)]
    assert images[0].image == 'diagram.png'
    assert images[1].image.startswith('data:image/jpeg;base64,')
    assert images[2].image == 'last-supported.png'
