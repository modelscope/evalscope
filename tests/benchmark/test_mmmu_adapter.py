from io import BytesIO
from PIL import Image

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage
from evalscope.benchmarks.mmmu.mmmu_adapter import MULTI_CHOICE_TYPE, MMMUAdapter
from evalscope.config import TaskConfig


def _adapter() -> MMMUAdapter:
    return MMMUAdapter(
        benchmark_meta=BenchmarkMeta(name='mmmu', dataset_id='AI-ModelScope/MMMU', eval_split='validation'),
        task_config=TaskConfig(datasets=['mmmu']),
    )


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format=image_format)
    return buffer.getvalue()


def test_mmmu_preserves_placeholder_order_and_detects_mime_type() -> None:
    content, answers = _adapter().create_content_and_answers_list({
        'question_type': MULTI_CHOICE_TYPE,
        'question': 'Compare <image 1> with <image 2>.',
        'options': "['A', 'B']",
        'image_1': {'bytes': _image_bytes('PNG')},
        'image_2': {'bytes': _image_bytes('JPEG')},
    })

    images = [item for item in content if isinstance(item, ContentImage)]
    assert [image.image.split(';', 1)[0] for image in images] == ['data:image/png', 'data:image/jpeg']
    assert answers == ['A', 'B']


def test_mmmu_supports_undecoded_path_image() -> None:
    content, _ = _adapter().create_content_and_answers_list({
        'question_type': MULTI_CHOICE_TYPE,
        'question': 'Read <image 1>.',
        'options': "['A', 'B']",
        'image_1': {'bytes': None, 'path': 'diagram.png'},
    })

    images = [item.image for item in content if isinstance(item, ContentImage)]
    assert images == ['diagram.png']
