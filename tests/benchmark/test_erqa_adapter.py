from io import BytesIO
from PIL import Image

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage, ContentText
from evalscope.benchmarks.erqa.erqa_adapter import ERQAAdapter
from evalscope.config import TaskConfig


def _adapter() -> ERQAAdapter:
    return ERQAAdapter(
        benchmark_meta=BenchmarkMeta(name='erqa', dataset_id='evalscope/ERQA', eval_split='test'),
        task_config=TaskConfig(datasets=['erqa']),
    )


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format='JPEG')
    return buffer.getvalue()


def test_erqa_normalizes_plural_images() -> None:
    sample = _adapter().record_to_sample({
        'question': 'Where should the robot move?',
        'answer': 'A',
        'question_type': 'Spatial Reasoning',
        'images': [{'path': 'first.png'}, {'bytes': _image_bytes()}],
    })

    content = sample.input[0].content
    assert isinstance(content[0], ContentText)
    images = [item for item in content if isinstance(item, ContentImage)]
    assert images[0].image == 'first.png'
    assert images[1].image.startswith('data:image/jpeg;base64,')
