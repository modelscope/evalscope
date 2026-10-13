# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the AI2D adapter's record conversion and dataset binding."""
import io

from PIL import Image

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _adapter():
    task_cfg = TaskConfig(model='mock', datasets=['ai2d'])
    return get_benchmark('ai2d', task_cfg)


def _record(image_format: str = 'JPEG') -> dict:
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buffer, format=image_format)
    return {
        'question': 'What is the first stage in the cycle?',
        'options': ['fruit', 'fruit development', 'flowering', 'transplant'],
        'answer': 3,
        'image': {'bytes': buffer.getvalue(), 'path': '1287.png'},
        'abcLabel': False,
    }


def test_dataset_binding_pins_per_question_image_semantics() -> None:
    """v1.1 switched to evalscope/ai2d with per-question images; scores across versions differ."""
    meta = _adapter()._benchmark_meta
    assert meta.dataset_id == 'evalscope/ai2d'
    assert meta.evaluation_version == 'v1.1'


def test_jpeg_bytes_get_jpeg_mime_header() -> None:
    """The dataset stores JPEG bytes; a fixed png header used to mislabel every image."""
    sample = _adapter().record_to_sample(_record('JPEG'))
    image = next(c for c in sample.input[0].content if hasattr(c, 'image'))
    assert image.image.startswith('data:image/jpeg;base64,')


def test_png_bytes_get_png_mime_header() -> None:
    sample = _adapter().record_to_sample(_record('PNG'))
    image = next(c for c in sample.input[0].content if hasattr(c, 'image'))
    assert image.image.startswith('data:image/png;base64,')


def test_answer_index_maps_to_letter_target_and_choices() -> None:
    sample = _adapter().record_to_sample(_record())
    assert sample.target == 'D'
    assert sample.choices == ['fruit', 'fruit development', 'flowering', 'transplant']
