# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the MMBench adapter's hint handling."""
import io

from PIL import Image

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _adapter():
    task_cfg = TaskConfig(model='mock', datasets=['mm_bench'])
    return get_benchmark('mm_bench', task_cfg)


def _record(hint: str) -> dict:
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2)).save(buffer, format='JPEG')
    return {
        'index': 293,
        'hint': hint,
        'question': 'Which material is this spatula made of?',
        'A': 'rubber',
        'B': 'cotton',
        'C': 'nan',
        'D': 'nan',
        'answer': 'A',
        'image': {'bytes': buffer.getvalue()},
    }


def _text(sample) -> str:
    return next(c for c in sample.input[0].content if hasattr(c, 'text')).text


def test_nan_hint_is_dropped() -> None:
    """lmms-lab/MMBench stores a missing hint as the string 'nan' (4,007 of 4,329 dev rows per subset)."""
    text = _text(_adapter().record_to_sample(_record('nan')))
    assert 'nan' not in text
    assert '\n\nWhich material is this spatula made of?\n\n' in text


def test_real_hint_is_separated_from_question() -> None:
    text = _text(_adapter().record_to_sample(_record('Figure: a spatula.')))
    assert 'Hint: Figure: a spatula.\nQuestion: Which material is this spatula made of?' in text


def test_evaluation_version_bumped_for_prompt_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'
