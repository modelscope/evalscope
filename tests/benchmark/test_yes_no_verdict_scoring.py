"""Regression tests: Yes/No benchmarks must not credit a reply via substring match.

Covers pope, hallusion_bench and drivel_binary, whose match_score previously did
``reference in filtered_prediction.strip().upper()`` and so credited a "Yes"/absent reply for a "NO" label
whenever the uppercased text contained the letters "no" ("know", "not", "no explanation", ...).
"""

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric.scorer import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

ADAPTERS = ['pope', 'hallusion_bench', 'drivel_binary']


def _sample_score(name: str, prediction: str, label: str) -> SampleScore:
    adapter = get_benchmark(name, TaskConfig(model='m', datasets=[name]))
    sample = Sample(id=0, input='Q:', target=label.upper(), metadata={'answer': label.lower()})
    state = TaskState(model='m', sample=sample, output=ModelOutput.from_content('m', prediction), completed=True)
    return adapter.calculate_metrics(state)


def _score(name: str, prediction: str, label: str) -> float:
    return _sample_score(name, prediction, label).score.main_value


@pytest.mark.parametrize('name', ADAPTERS)
@pytest.mark.parametrize(
    ('prediction', 'label'),
    [
        ('Yes, I know it.', 'No'),
        ('I do not know.', 'No'),
        ('Yes, no explanation needed.', 'No'),
    ],
)
def test_yes_or_absent_verdict_is_not_credited_for_a_no_label(name: str, prediction: str, label: str) -> None:
    assert _score(name, prediction, label) == 0


@pytest.mark.parametrize('name', ADAPTERS)
@pytest.mark.parametrize(
    ('prediction', 'label', 'expected'),
    [
        ('Yes', 'Yes', 1),
        ('No', 'No', 1),
        ('No.', 'No', 1),
        ('YES', 'Yes', 1),
        ('yes', 'Yes', 1),
        ('no', 'No', 1),
        ('no.', 'No', 1),
        ('Yes', 'No', 0),
        ('No', 'Yes', 0),
    ],
)
def test_verdict_matches_label(name: str, prediction: str, label: str, expected: int) -> None:
    assert _score(name, prediction, label) == expected


@pytest.mark.parametrize('name', ['pope', 'drivel_binary'])
def test_ambiguous_reply_does_not_create_a_false_positive(name: str) -> None:
    adapter = get_benchmark(name, TaskConfig(model='m', datasets=[name]))
    scores = [_sample_score(name, 'Yes', 'Yes'), _sample_score(name, 'I do not know.', 'No')]

    metrics = {item.metric_name: item.score for item in adapter.aggregate_scores(scores)}

    assert metrics['accuracy'] == 0.5
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['yes_ratio'] == 0.5


@pytest.mark.parametrize('name', ['pope', 'drivel_binary'])
def test_ambiguous_yes_reply_counts_as_a_miss(name: str) -> None:
    adapter = get_benchmark(name, TaskConfig(model='m', datasets=[name]))
    scores = [_sample_score(name, 'I do not know.', 'Yes')]

    metrics = {item.metric_name: item.score for item in adapter.aggregate_scores(scores)}

    assert metrics['accuracy'] == 0.0
    assert metrics['recall'] == 0.0
    assert metrics['yes_ratio'] == 0.0
