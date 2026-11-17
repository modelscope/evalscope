"""A zero native evaluation limit must not start dataset loading or inference."""

import json
from pathlib import Path
from typing import Any

import pytest

from evalscope import TaskConfig, run_task
from evalscope.benchmarks.general_mcq.general_mcq_adapter import GeneralMCQAdapter
from evalscope.models.mockllm import MockLLM


@pytest.mark.parametrize('limit', [0, 0.0, '0', '0.0'])
def test_zero_limit_skips_loading_and_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: Any) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail('A zero-limit evaluation must not load a dataset or generate predictions')

    monkeypatch.setattr(GeneralMCQAdapter, 'load_dataset', forbidden)
    monkeypatch.setattr(MockLLM, 'generate', forbidden)
    config = TaskConfig(
        model='zero-limit',
        eval_type='mock_llm',
        datasets=['general_mcq'],
        limit=limit,
        repeats=3,
        work_dir=str(tmp_path / 'run'),
        no_timestamp=True,
    )

    assert run_task(config)['general_mcq'] == {}
    assert not list((tmp_path / 'run' / 'predictions').rglob('*.jsonl'))


@pytest.mark.parametrize('limit,expected', [(None, 4), (0.01, 0), (0.5, 2), (2, 2)])
def test_native_limit_controls_prediction_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: float | int | None, expected: int
) -> None:
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    rows = [{'question': f'Question {index}', 'A': 'first', 'B': 'second', 'answer': 'B'} for index in range(4)]
    (data_dir / 'default_val.jsonl').write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    monkeypatch.setattr(MockLLM, 'default_output', 'ANSWER: B')
    monkeypatch.setattr('evalscope.report.gen_html_report_file', lambda *args, **kwargs: '')
    original_generate = MockLLM.generate
    calls = []

    def counted_generate(self: MockLLM, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original_generate(self, *args, **kwargs)

    monkeypatch.setattr(MockLLM, 'generate', counted_generate)
    config = TaskConfig(
        model='limit-control',
        eval_type='mock_llm',
        datasets=['general_mcq'],
        dataset_args={'general_mcq': {'local_path': str(data_dir), 'subset_list': ['default']}},
        limit=limit,
        eval_batch_size=1,
        work_dir=str(tmp_path / 'run'),
        no_timestamp=True,
    )

    report = run_task(config)['general_mcq']

    assert len(calls) == expected
    if expected:
        assert report.score == 1.0
        predictions = list((tmp_path / 'run' / 'predictions').rglob('*.jsonl'))
        assert len(predictions) == 1
        assert len(predictions[0].read_text().splitlines()) == expected
    else:
        assert report == {}
