import pytest
from types import SimpleNamespace
from unittest import mock

from evalscope.api.metric import MetricSelector, SampleScore, Score
from evalscope.benchmarks.omnidoc_bench.legacy.omnidoc_bench_adapter import OmniDocBenchAdapter


def test_legacy_omnidoc_aggregates_canonical_metrics() -> None:
    selector = MetricSelector(name='normalized_score', aggregation='macro_mean')
    adapter = OmniDocBenchAdapter.__new__(OmniDocBenchAdapter)
    adapter._benchmark_meta = SimpleNamespace(metric_list=[], primary_metric=selector)
    adapter.match_method = 'quick_match'
    sample_scores = [SampleScore(sample_id=1, score=Score(prediction='markdown', metadata={'reference': {}}))]
    raw_scores = {
        'text_block_Edit_dist_EN': 0.2,
        'table_TEDS_CH': 0.8,
        'overall_EN': 0.7,
        'overall_CH': 0.9,
    }

    with mock.patch(
        'evalscope.benchmarks.omnidoc_bench.legacy.end2end_eval.End2EndEvaluator.score',
        return_value=raw_scores,
    ):
        aggregated = adapter.aggregate_scores(sample_scores)

    identities = [score.identity for score in aggregated]
    assert any(
        identity.name == 'text_block_edit_dist' and identity.dimensions == {'language': 'en'} for identity in identities
    )
    assert any(identity.name == 'table_teds' and identity.dimensions == {'language': 'ch'} for identity in identities)
    primary_matches = [score for score in aggregated if selector.matches(score.identity)]
    assert len(primary_matches) == 1
    assert primary_matches[0].score == pytest.approx(0.8)
