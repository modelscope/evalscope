from io import BytesIO
from PIL import Image
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage
from evalscope.benchmarks.mmmu_pro.mmmu_pro_adapter import MMMUPROAdapter
from evalscope.config import TaskConfig


def _adapter(dataset_format: str) -> MMMUPROAdapter:
    adapter = MMMUPROAdapter(
        benchmark_meta=BenchmarkMeta(name='mmmu_pro', dataset_id='AI-ModelScope/MMMU_Pro', eval_split='test'),
        task_config=TaskConfig(datasets=['mmmu_pro']),
    )
    adapter.dataset_format = dataset_format
    return adapter


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
        image_1={'bytes': _image_bytes('PNG')},
        image_2={'bytes': _image_bytes('JPEG')},
    ))

    images = [item for item in sample.input[0].content if isinstance(item, ContentImage)]
    assert [image.image.split(';', 1)[0] for image in images] == ['data:image/png', 'data:image/jpeg']


def test_mmmu_pro_vision_format_supports_path_image() -> None:
    sample = _adapter('vision').record_to_sample(_record(image={'bytes': None, 'path': 'question.png'}))

    images = [item.image for item in sample.input[0].content if isinstance(item, ContentImage)]
    assert images == ['question.png']
