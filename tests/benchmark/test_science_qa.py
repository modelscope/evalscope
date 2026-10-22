# Copyright (c) Alibaba, Inc. and its affiliates.
"""ScienceQA must show the record's context (`hint`) to the model, as the official prompt does."""
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def _adapter():
    return get_benchmark('science_qa', TaskConfig(datasets=['science_qa']))


def _record(hint: str) -> dict:
    return {
        'question': 'Which text structure does the text use?',
        'hint': hint,
        'choices': ['problem-solution', 'cause-effect'],
        'answer': 1,
        'image': None,
    }


def _text(sample) -> str:
    return sample.input[0].content[0].text


def test_context_is_included_before_the_question() -> None:
    text = _text(_adapter().record_to_sample(_record('Read the text.\nWatching TV results in too much sitting.')))
    assert 'Context: Read the text.\nWatching TV results in too much sitting.\nWhich text structure' in text


def test_empty_context_is_omitted() -> None:
    text = _text(_adapter().record_to_sample(_record('')))
    assert 'Context' not in text
    assert '\n\nWhich text structure does the text use?\n\n' in text


def test_evaluation_version_bumped_for_prompt_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'
