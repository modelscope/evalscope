"""Second-round regressions for the CI metadata PR (review of 717c69a5).

Each test maps to a reviewer-reproduced issue: real repeat loading, string
categories, unit-invariance of the width flag, and the vocabulary contract.
"""
import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.metrics.aggregators.aggregators import MeanAndCI


def repeat_scores(questions: int, repeats: int):
    """Mimic the real repeat loader: distinct sample_id per generation, shared
    group_id per question, generation_index 0..repeats-1 (issue repro)."""
    scores = []
    for q in range(questions):
        for r in range(repeats):
            scores.append(
                SampleScore(
                    sample_id=f'q{q}-g{r}',
                    group_id=f'q{q}',
                    generation_index=r,
                    score=Score(value={'accuracy': 1.0 if (q + r) % 2 == 0 else 0.0}),
                )
            )
    return scores


def test_real_repeat_count_is_rejected_at_construction():
    # Config-driven (review: do not infer repeats from data shape — a dataset where one
    # item yields several scores is legitimate). The repeat count is known at
    # construction, same as bayes_at_n.
    with pytest.raises(ValueError, match='does not support repeats'):
        MeanAndCI(repeats=100)
    with pytest.raises(ValueError, match='does not support repeats'):
        MeanAndCI(repeats=2)

    # hand-built repeat-shaped data must still be caught by the duplicate-id check
    with pytest.raises(ValueError, match='more than once'):
        MeanAndCI()(
            [SampleScore(sample_id='dup', score=Score(value={'accuracy': 1.0})) for _ in range(2)]
        )


def test_plain_scores_without_repeat_structure_pass():
    scores = [SampleScore(sample_id=f's-{i}', score=Score(value={'accuracy': float(i % 2)})) for i in range(50)]
    entries = MeanAndCI()(scores)
    assert len(entries) == 1 and entries[0].aggregation == 'mean'


def test_string_category_is_not_split_into_characters():
    from types import SimpleNamespace

    from evalscope.api.metric import AggScore, MetricSelector
    from evalscope.metrics.aggregators.aggregators import MeanAndCI
    from evalscope.report.generator import ReportGenerator

    scores = [SampleScore(sample_id='s-1', score=Score(value={'accuracy': 1.0}))]
    subset_scores = {'dataset': MeanAndCI()(scores)}
    adapter = SimpleNamespace(
        name='ceval',
        pretty_name='CEval',
        description='',
        category_map={'dataset': 'STEM'},  # real C-Eval shape: a plain string
        primary_metric=MetricSelector(name='accuracy'),
    )
    report = ReportGenerator.generate_report(subset_scores, 'model', adapter)

    category = report.metrics[0].categories[0]
    assert category.name == ('STEM',)  # one-element tuple, never ('S','T','E','M')
    subset = report.metrics[0].categories[0].subsets[0]
    assert subset.ci is not None and subset.ci.low is not None


def test_width_flag_is_unit_invariant():
    # 100 alternating 3/4 scores: bootstrap interval [3.4, 3.6] must NOT be flagged.
    values = [3.0 if i % 2 == 0 else 4.0 for i in range(100)]
    scores = [SampleScore(sample_id=f's-{i}', score=Score(value={'judge_score': v})) for i, v in enumerate(values)]
    ci = MeanAndCI()(scores)[0]
    assert (ci.metadata['ci']['flags']) == [] or 'wide_interval' not in ci.metadata['ci']['flags']

    # dividing by 1000 must not change the assessment either
    scaled = [v / 1000 for v in values]
    scores_scaled = [SampleScore(sample_id=f's-{i}', score=Score(value={'judge_score': v})) for i, v in enumerate(scaled)]
    ci_scaled = MeanAndCI()(scores_scaled)[0]
    assert 'wide_interval' not in ci_scaled.metadata['ci']['flags']


def test_low_confidence_shows_interval_alongside_flag():
    from evalscope.report.generator import _format_ci

    rendered = _format_ci({'low': 0.6397, 'high': 0.9476, 'flags': ['low_confidence']}, 20, None)
    assert 'low confidence' in rendered
    assert '[' in rendered and ']' in rendered  # interval shown alongside, not replaced


def test_mean_and_ci_removed_from_known_aggregations():
    from evalscope.api.metric.semantics import KNOWN_AGGREGATIONS

    assert 'mean_and_ci' not in KNOWN_AGGREGATIONS


def test_config_boundary_rejects_repeats():
    from unittest.mock import patch

    from evalscope.benchmarks.gsm8k.gsm8k_adapter import GSM8KAdapter
    from evalscope.config import TaskConfig

    adapter = GSM8KAdapter(
        benchmark_meta=__import__('evalscope.api.registry', fromlist=['get_benchmark']).get_benchmark('gsm8k').benchmark_meta
    )
    adapter._task_config = TaskConfig(repeats=5)
    # `aggregation` is a read-through of the meta; the boundary must see the configured value
    adapter.benchmark_meta.aggregation = 'mean_and_ci'
    with pytest.raises(ValueError, match='does not support repeats'):
        adapter.aggregate_scores([])
