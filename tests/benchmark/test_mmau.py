# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for MMAU gold mapping and answer-letter parsing."""
from pathlib import Path

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.mmau.mmau_adapter import MMAUAdapter
from evalscope.config import TaskConfig

CHOICES = ['dog', 'cat', 'cow', 'horse']


def _adapter() -> MMAUAdapter:
    return get_benchmark('mmau', TaskConfig(datasets=['mmau']))


def _record(answer: str) -> dict:
    return {
        'question': 'What animal is heard?',
        'choices': CHOICES,
        'answer': answer,
        'audio': {'bytes': b'RIFF0000WAVE'},
        'task': 'sound',
    }


def _score(prediction: str, target: str) -> Score:
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target=target, choices=CHOICES, metadata={'task': 'sound'}),
        output=ModelOutput.from_content('mock', prediction),
        completed=True,
    )
    return _adapter().calculate_metrics(state).score


def test_unlabeled_test_split_is_not_scored() -> None:
    """lmms-lab/mmau's 9,000-row ``test`` split stores 'NA' as every answer; it must not become gold 'A'."""
    adapter = _adapter()
    assert adapter.subset_list == ['test_mini']
    with pytest.raises(ValueError, match='not one of the choices'):
        adapter.record_to_sample(_record('NA'))
    assert adapter.record_to_sample(_record('Cow')).target == 'C'


@pytest.mark.parametrize('prediction', ['The sound is a dog barking.', 'I cannot tell from this audio.'])
def test_unparsed_answer_is_not_credited_as_a(prediction: str) -> None:
    score = _score(prediction, 'A')
    assert score.extracted_prediction == ''
    assert score.main_value == 0.0


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [
        ('ANSWER: Dog barking', ''),
        ('ANSWER: a dog barking', ''),
        ('The answer: a cow? No.\nANSWER: B', 'B'),
        ('**Answer**: b.', 'B'),
        ('ANSWER: [D]', 'D'),
        ('ANSWER: A dog barking', ''),
        ('ANSWER: C major', ''),
        ('ANSWER: B\nThe answer: A dog barking.', 'B'),
        ('ANSWER: B\nANSWER: A dog barking', 'B'),
        ('ANSWER: A\nANSWER: B', 'B'),
        ('ANSWER: A\nANSWER: [b]', 'B'),
        ('ANSWER: [B, C]', ''),
        ('B', 'B'),
        ('(d)', 'D'),
    ],
)
def test_answer_line_reads_only_a_choice_letter(prediction: str, expected: str) -> None:
    assert _score(prediction, 'B').extracted_prediction == expected


def test_evaluation_version_bumped_for_scoring_change() -> None:
    assert _adapter()._benchmark_meta.evaluation_version == 'v1.1'


def test_natural_language_does_not_earn_credit_for_a() -> None:
    score = _score('ANSWER: A dog barking', 'A')
    assert score.extracted_prediction == ''
    assert score.main_value == 0.0


def test_configured_filter_reaches_scoring() -> None:
    adapter = get_benchmark(
        'mmau',
        TaskConfig(
            datasets=['mmau'],
            dataset_args={'mmau': {'filters': {'remove_until': '</think>'}}},
        ),
    )
    state = TaskState(
        model='mock',
        sample=Sample(input='q', target='B', choices=CHOICES, metadata={'task': 'sound'}),
        output=ModelOutput.from_content('mock', 'ANSWER: A</think>B'),
        completed=True,
    )
    score = adapter.calculate_metrics(state).score
    assert score.extracted_prediction == 'B'
    assert score.main_value == 1.0


def test_generated_metadata_matches_runtime() -> None:
    from evalscope.service.utils.benchmarks import build_benchmark_entry

    entry = build_benchmark_entry('mmau')
    adapter = _adapter()
    assert entry['subset_list'] == adapter.subset_list == ['test_mini']
    assert entry['meta']['description'] == adapter.description
    for language in ('en', 'zh'):
        assert 'NA' in entry['description'][language]['full']


def test_report_uses_overall_accuracy(tmp_path: Path) -> None:
    from evalscope.api.metric import SampleScore

    adapter = _adapter()
    scores = [
        SampleScore(score=_score('ANSWER: B', 'B'), sample_metadata={'task': 'sound'}),
        SampleScore(score=_score('No answer.', 'A'), sample_metadata={'task': 'music'}),
    ]
    report = adapter.generate_report(
        {'test_mini': adapter.aggregate_scores(scores)}, model_name='mock', output_dir=str(tmp_path)
    )
    assert report.score == 0.5
    assert report.primary_metric_identity.dimensions == {'scope': 'overall'}
    assert len(report.metrics) == 3
