from io import BytesIO
from PIL import Image
from typing import Any, Dict

from evalscope.api.messages import ContentImage, ContentText
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.mmmu_pro.mmmu_pro_adapter import MMMUPROAdapter
from evalscope.config import TaskConfig


def _adapter(dataset_format: str) -> MMMUPROAdapter:
    config = TaskConfig(
        datasets=['mmmu_pro'],
        dataset_args={'mmmu_pro': {'extra_params': {'dataset_format': dataset_format}}},
    )
    return get_benchmark('mmmu_pro', config)


def _record(**overrides: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        'id': 'sample',
        'options': "['A', 'B']",
        'answer': 'A',
        'subject': 'Math',
        'question': 'Compare <image 1> with <image 2>.',
    }
    record.update(overrides)
    return record


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format=image_format)
    return buffer.getvalue()


def test_mmmu_pro_normalizes_indexed_images() -> None:
    sample = _adapter('standard (4 options)').record_to_sample(_record(
        question='Before <image 1> between <image 2> after.',
        image_1={'bytes': _image_bytes('PNG')},
        image_2={'bytes': _image_bytes('JPEG')},
    ))

    content = sample.input[0].content
    assert [item.type for item in content] == ['text', 'image', 'text', 'image', 'text']
    text_segments = [item.text for item in content if isinstance(item, ContentText)]
    assert text_segments[0].endswith('\n\nBefore ')
    assert text_segments[1] == ' between '
    assert text_segments[2] == ' after.\n\nA) A\nB) B'
    images = [item for item in sample.input[0].content if isinstance(item, ContentImage)]
    assert [image.image.split(';', 1)[0] for image in images] == ['data:image/png', 'data:image/jpeg']


def test_mmmu_pro_vision_format_supports_path_image() -> None:
    adapter = _adapter('vision')
    sample = adapter.record_to_sample(_record(image={'bytes': None, 'path': 'question.png'}))

    assert adapter.dataset_format == 'vision'
    assert adapter.default_subset == 'vision'
    assert [item.type for item in sample.input[0].content] == ['text', 'image']
    images = [item.image for item in sample.input[0].content if isinstance(item, ContentImage)]
    assert images == ['question.png']
