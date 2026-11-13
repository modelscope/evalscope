"""BBEH splits tasks into subsets, scores with the official fuzzy match and reports the adjusted harmonic mean."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

RECORD = {'task': 'boardgame qa', 'input': 'Is the statement proved?', 'target': 'disproved', 'canary': '', 'mini': 1}


def _adapter(extra_params: Optional[Dict[str, Any]] = None) -> DataAdapter:
    dataset_args = {'bbeh': {'extra_params': extra_params}} if extra_params else {}
    return get_benchmark('bbeh', TaskConfig(model='mock', datasets=['bbeh'], dataset_args=dataset_args))


def test_record_to_sample_uses_task_as_subset() -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample(RECORD)
    assert sample.subset_key == 'boardgame_qa'
    assert sample.subset_key in adapter.subset_list
    assert sample.target == 'disproved'
    assert sample.metadata == {'mini': True}
    prompt = adapter.format_prompt_template(sample)
    assert prompt.startswith('Is the statement proved?\n\nThink step by step')
    assert prompt.endswith('For example, "The answer is: (a)".')


@pytest.mark.parametrize(
    ('completion', 'reference', 'correct'),
    [
        # The examples at the end of the official bbeh/evaluate.py.
        ('Ok The final answer is: \\boxed{4}.', '4', True),
        ('[Reasoning] The final answer is: \\boxed{4}.', '3', False),
        ('Alright! The final answer is: 2, 3, 4', '2,3,4', True),
        ('blah blah The final answer is: 2, 3, 4', '2,3,5', False),
        ('Ok The answer is: (A)', 'a', True),
        ('Ok The answer is: (A)', 'b', False),
        ("Ok The answer is: **25**\nHere's why.", '25.0', True),
        ("Ok The answer is: **25**\nHere's why.", '26.0', False),
    ],
)
def test_scores_like_official_evaluate(completion: str, reference: str, correct: bool) -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({**RECORD, 'target': reference})
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=completion))
    extracted = adapter.extract_answer(completion, state)
    score = adapter.match_score(completion, extracted, reference, state)
    assert score.main_value == float(correct)


def _write_dataset(path: Path) -> None:
    path.mkdir()
    (path / 'README.md').write_text(
        '---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: train.jsonl\n---\n'
    )
    rows = [
        # boardgame qa: 2 of 3 answered correctly by the mock reply, 2 in BBEH Mini.
        {**RECORD, 'target': 'unknown', 'mini': 1},
        {**RECORD, 'target': 'unknown', 'mini': 0},
        {**RECORD, 'target': 'proved', 'mini': 1},
        # web of lies: none answered correctly.
        {**RECORD, 'task': 'web of lies', 'target': 'yes', 'mini': 1},
    ]
    (path / 'train.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_params: Dict[str, Any]) -> Dict[str, Any]:
    from evalscope import run_task
    from evalscope.models.mockllm import MockLLM

    data = tmp_path / 'bbeh'
    _write_dataset(data)
    monkeypatch.setattr(MockLLM, 'default_output', 'Let me think. The answer is: unknown.')
    work = tmp_path / 'run'
    run_task(
        TaskConfig(
            model='offline-bbeh',
            eval_type='mock_llm',
            datasets=['bbeh'],
            dataset_args={'bbeh': {'local_path': str(data), 'extra_params': extra_params}},
            no_timestamp=True,
            work_dir=str(work),
        )
    )
    report = json.loads((work / 'reports/offline-bbeh/bbeh.json').read_text())
    return {(m['identity']['aggregation']): m for m in report['metrics']}


def test_offline_run_reports_micro_average_and_harmonic_mean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _run(tmp_path, monkeypatch, {})

    accuracy = metrics['mean']
    assert (accuracy['score'], accuracy['num']) == (0.5, 4)
    subsets = {s['name']: s['score'] for c in accuracy['categories'] for s in c['subsets']}
    assert subsets == pytest.approx({'boardgame_qa': 2 / 3, 'web_of_lies': 0.0}, abs=1e-4)

    # Harmonic mean of (accuracy in percent + 1) over tasks, as in the BBEH paper.
    expected = 2 / (1 / (100 * 2 / 3 + 1) + 1 / (0 + 1)) / 100
    assert metrics['harmonic_mean']['score'] == pytest.approx(expected, abs=1e-4)


def test_offline_run_mini_keeps_only_mini_examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _run(tmp_path, monkeypatch, {'mini': True})

    accuracy = metrics['mean']
    assert accuracy['num'] == 3
    subsets = {s['name']: (s['score'], s['num']) for c in accuracy['categories'] for s in c['subsets']}
    assert subsets == {'boardgame_qa': (0.5, 2), 'web_of_lies': (0.0, 1)}
