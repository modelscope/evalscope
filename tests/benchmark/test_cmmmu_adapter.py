import base64
import pytest
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.messages import ContentImage, ContentText
from evalscope.benchmarks.cmmmu.cmmmu_adapter import CMMMUAdapter
from evalscope.config import TaskConfig


@pytest.fixture
def adapter() -> CMMMUAdapter:
    return CMMMUAdapter(
        benchmark_meta=BenchmarkMeta(name='cmmmu', dataset_id='dummy', eval_split='val'),
        task_config=TaskConfig(datasets=['cmmmu']),
    )


def _record(question: str, **overrides: Any) -> Dict[str, Any]:
    record = {
        'question': question,
        'type': '选择',
        'option1': '甲',
        'option2': '乙',
        'option3': '丙',
        'option4': '丁',
        **{f'image_{index}_filename': '' for index in range(1, CMMMUAdapter.MAX_IMAGES + 1)},
    }
    record.update(overrides)
    return record


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='white').save(buffer, format=image_format)
    return buffer.getvalue()


def test_create_content_list_supports_undecoded_path_image(adapter: CMMMUAdapter, tmp_path: Path) -> None:
    image_path = tmp_path / 'diagram.png'
    Image.new('RGB', (2, 2), color='white').save(image_path)
    record = _record(
        '观察<img="diagram.png">，选择正确答案。',
        image_1={'path': str(image_path), 'bytes': None},
        image_1_filename='diagram.png',
    )

    content_list = adapter.create_content_list(record)

    images = [content for content in content_list if isinstance(content, ContentImage)]
    assert [content.image for content in images] == [str(image_path)]


def test_create_content_list_preserves_image_order_and_detects_mime_type(adapter: CMMMUAdapter) -> None:
    png_bytes = _image_bytes('PNG')
    jpeg_bytes = _image_bytes('JPEG')
    record = _record(
        '比较<img="left.png">与<img="right.jpg">，选择正确答案。',
        image_1={'path': 'left.png', 'bytes': png_bytes},
        image_1_filename='left.png',
        image_2={'path': 'right.jpg', 'bytes': jpeg_bytes},
        image_2_filename='right.jpg',
    )

    content_list = adapter.create_content_list(record)

    assert [type(content) for content in content_list] == [
        ContentText,
        ContentImage,
        ContentText,
        ContentImage,
        ContentText,
    ]
    images = [content for content in content_list if isinstance(content, ContentImage)]
    expected_png = f'data:image/png;base64,{base64.b64encode(png_bytes).decode()}'
    assert images[0].image == expected_png
    assert images[1].image.startswith('data:image/jpeg;base64,')
