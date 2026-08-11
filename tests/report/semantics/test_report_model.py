import pytest
from typing import Dict, List, Optional

from evalscope.api.metric import AggScore
from evalscope.api.metric.semantics import MetricIdentity, MetricRole, MetricSelector
from evalscope.report.generator import ReportGenerator
from evalscope.report.report import Report


class _StubAdapter:

    def __init__(self, name: str, selector: Optional[MetricSelector] = None) -> None:
        self.name = name
        self.primary_metric = selector
        self.category_map = {}
        self.description = ''
        self.pretty_name = name


def _scores() -> Dict[str, List[AggScore]]:
    return {
        'test': [
            AggScore(score=0.8, metric_name='f1', aggregation='mean', num=10),
            AggScore(score=0.75, metric_name='precision', aggregation='mean', num=10),
        ]
    }


def _report() -> Report:
    return ReportGenerator.generate_report(
        _scores(),
        'model',
        _StubAdapter('conll2003', MetricSelector(name='f1', aggregation='mean')),
    )


def test_generator_groups_by_identity_and_selects_primary() -> None:
    report = _report()
    assert [metric.identity.name for metric in report.metrics] == ['f1', 'precision']
    assert report.primary_metric_identity == MetricIdentity(name='f1', aggregation='mean')
    assert [metric.role for metric in report.metrics] == [MetricRole.PRIMARY, MetricRole.AUXILIARY]


def test_generator_preserves_scores() -> None:
    report = _report()
    assert [metric.score for metric in report.metrics] == [0.8, 0.75]


def test_v2_serialization_contains_no_v1_metric_fields() -> None:
    data = _report().to_dict()
    assert data['schema_version'] == 2
    assert 'score' not in data
    assert 'primary_metric_name' not in data
    assert data['primary_metric_identity']['name'] == 'f1'
    for metric in data['metrics']:
        assert set(('name', 'semantic_id')).isdisjoint(metric)
        assert set(('identity', 'semantics')).issubset(metric)


def test_v2_round_trip_uses_persisted_semantics_without_resolution() -> None:
    original = _report()
    restored = Report.from_dict(original.to_dict())
    assert restored == original


def test_v1_report_migrates_without_changing_values() -> None:
    report = Report.from_dict({
        'metric_schema_version': 1,
        'dataset_name': 'conll2003',
        'primary_metric_name': 'mean_f1_score',
        'score': 0.8,
        'metrics': [{
            'name': 'mean_f1_score',
            'score': 0.8,
            'semantic_id': 'quality.f1.ratio',
            'categories': [],
        }],
    })
    assert report.metrics[0].score == 0.8
    assert report.metrics[0].identity == MetricIdentity(name='f1', aggregation='mean')
    assert report.metrics[0].semantics.role is MetricRole.PRIMARY
    assert 'name' not in report.to_dict()['metrics'][0]


def test_unknown_legacy_metric_is_preserved_as_diagnostic() -> None:
    report = Report.from_dict({
        'dataset_name': 'third_party',
        'metrics': [{
            'name': 'vendor_metric',
            'score': 3.5,
            'categories': [],
        }],
    })
    metric = report.metrics[0]
    assert metric.legacy_name == 'vendor_metric'
    assert metric.score == 3.5
    assert metric.role is MetricRole.DIAGNOSTIC
    assert report.primary_metric is None


def test_unknown_legacy_metric_uses_persisted_semantic_anchor() -> None:
    report = Report.from_dict({
        'dataset_name': 'third_party',
        'metrics': [{
            'name': 'vendor_metric',
            'score': 0.75,
            'semantic_id': 'quality.accuracy.ratio',
            'categories': [],
        }],
    })

    metric = report.metrics[0]
    assert metric.identity == MetricIdentity(name='vendor_metric', aggregation='identity')
    assert metric.score == 0.75
    assert metric.semantics.semantic_id == 'quality.accuracy.ratio'
    assert metric.role is MetricRole.PRIMARY
    assert metric.legacy_name is None


def test_unknown_noncanonical_legacy_name_uses_isolated_diagnostic_identity() -> None:
    report = Report.from_dict({
        'dataset_name': 'third_party',
        'metrics': [{
            'name': 'Vendor Metric @ 1',
            'score': 3.5,
            'categories': [],
        }],
    })

    metric = report.metrics[0]
    assert metric.identity == MetricIdentity(
        name='legacy_metric', aggregation='identity', dimensions={'original_name': 'Vendor Metric @ 1'}
    )
    assert metric.legacy_name == 'Vendor Metric @ 1'
    assert metric.role is MetricRole.DIAGNOSTIC


def test_unknown_valid_third_party_identity_can_be_written_as_diagnostic() -> None:
    report = ReportGenerator.generate_report(
        {'test': [AggScore(score=3.5, metric_name='vendor_metric', aggregation='mean', num=1)]},
        'model',
        _StubAdapter('third_party'),
    )
    assert report.metrics[0].identity.name == 'vendor_metric'
    assert report.metrics[0].role is MetricRole.DIAGNOSTIC
    assert report.primary_metric_identity is None


def test_multi_scored_report_without_selector_fails() -> None:
    with pytest.raises(ValueError, match='declare BenchmarkMeta.primary_metric'):
        ReportGenerator.generate_report(_scores(), 'model', _StubAdapter('conll2003'))


def test_selector_must_match_exactly_one_identity() -> None:
    scores = {
        'test': [
            AggScore(
                score=0.8,
                metric_name='accuracy',
                aggregation='mean',
                dimensions={'scope': 'a'},
                num=1,
            ),
            AggScore(
                score=0.9,
                metric_name='accuracy',
                aggregation='mean',
                dimensions={'scope': 'b'},
                num=1,
            ),
        ]
    }
    with pytest.raises(ValueError, match='matched 2 identities'):
        ReportGenerator.generate_report(scores, 'model', _StubAdapter('benchmark', MetricSelector(name='accuracy')))
