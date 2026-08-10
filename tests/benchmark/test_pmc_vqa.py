"""Unit tests for PMC-VQA sample construction.

The dataset stores each option with a redundant letter prefix (' B:Magnetic resonance
imaging '). Leaving it in place would render options as 'B) B:Magnetic resonance imaging',
which is a silent failure: the run still completes and only the answer distribution shifts.
"""
import io
import zipfile
from typing import Dict

from evalscope.api.dataset import Sample
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.pmc_vqa.pmc_vqa_adapter import strip_choice_prefix

RECORD: Dict[str, str] = {
    'Figure_path': 'PMC1_F1.jpg',
    'Question': ' What imaging technique is used? ',
    'Answer': 'Magnetic resonance imaging',
    'Choice A': ' A:X-ray ',
    'Choice B': ' B:Magnetic resonance imaging ',
    'Choice C': ' C: Computed tomography ',
    'Choice D': 'D:Ultrasound',
    'Answer_label': 'B',
}

# Smallest possible JPEG payload; only its bytes matter, it is never decoded here.
JPEG_BYTES = b'\xff\xd8\xff\xd9'


def test_strip_choice_prefix() -> None:
    assert strip_choice_prefix(' A:X-ray ', 'A') == 'X-ray'
    assert strip_choice_prefix(' C: Computed tomography ', 'C') == 'Computed tomography'
    assert strip_choice_prefix('D:Ultrasound', 'D') == 'Ultrasound'
    # Only the option's own letter is stripped, so answer text starting with another
    # letter's name survives intact.
    assert strip_choice_prefix('A:B:cell lymphoma', 'A') == 'B:cell lymphoma'
    assert strip_choice_prefix('No prefix at all', 'B') == 'No prefix at all'


def _build_sample() -> Sample:
    """Build a sample the way load() does: the archive stays open only while reading."""
    adapter = get_benchmark('pmc_vqa')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(f'images/{RECORD["Figure_path"]}', JPEG_BYTES)
    with zipfile.ZipFile(buffer) as archive:
        adapter._image_archive = archive
        return adapter.record_to_sample(RECORD)


def test_record_to_sample_builds_multimodal_mcq() -> None:
    sample = _build_sample()

    assert sample.choices == [
        'X-ray',
        'Magnetic resonance imaging',
        'Computed tomography',
        'Ultrasound',
    ]
    assert sample.target == 'B'
    assert sample.metadata == {'figure_path': 'PMC1_F1.jpg'}

    image, text = sample.input[0].content
    assert image.image.startswith('data:image/jpeg;base64,')
    assert 'What imaging technique is used?' in text.text
    assert "'ANSWER: [LETTER]'" in text.text


def test_rendered_options_are_not_prefixed_twice() -> None:
    text = _build_sample().input[0].content[1].text

    assert 'B) Magnetic resonance imaging' in text
    for letter in 'ABCD':
        assert f'{letter}) {letter}:' not in text
