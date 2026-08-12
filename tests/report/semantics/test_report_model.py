import pandas as pd
import pytest
from typing import Dict, List, Optional

from evalscope.api.metric import AggScore
from evalscope.api.metric.semantics import MetricIdentity, MetricRole, MetricSelector
from evalscope.metrics.semantics.entry import MetricEntry
from evalscope.metrics.semantics.resolver import SemanticsResolver
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


def test_collection_report_keeps_mixed_metrics_auxiliary() -> None:
    report = ReportGenerator.gen_collection_report(
        pd.DataFrame([
            {
                'metric': 'accuracy',
                'categories': 'default',
                'dataset_name': 'gsm8k',
                'subset_name': 'test',
                'score': 0.8,
            },
            {
                'metric': 'f1',
                'categories': 'default',
                'dataset_name': 'conll2003',
                'subset_name': 'test',
                'score': 0.7,
            },
        ]),
        'data_collection',
        'model',
    )

    assert [metric.identity.name for metric in report.metrics] == ['accuracy', 'f1']
    assert [metric.role for metric in report.metrics] == [MetricRole.AUXILIARY, MetricRole.AUXILIARY]
    assert report.primary_metric is None
    assert report.primary_metric_identity is None


def test_report_score_compatibility_prefers_primary_then_first_metric() -> None:
    report = _report()
    assert report.score == 0.8

    collection = ReportGenerator.gen_collection_report(
        pd.DataFrame([
            {
                'metric': 'accuracy',
                'categories': 'default',
                'dataset_name': 'gsm8k',
                'subset_name': 'test',
                'score': 0.6,
            }
        ]),
        'data_collection',
        'model',
    )
    assert collection.score == 0.6
    assert Report().score == 0.0
    assert 'score' not in report.to_dict()


def test_generator_accepts_an_injected_semantics_resolver() -> None:
    resolver = SemanticsResolver(
        metric_definitions={'vendor_metric': MetricEntry(baseline='quality.accuracy.ratio')},
        aggregation_semantics={},
        benchmark_overrides={},
    )
    report = ReportGenerator.generate_report(
        {'test': [AggScore(score=0.75, metric_name='vendor_metric', aggregation='mean', num=1)]},
        'model',
        _StubAdapter('third_party', MetricSelector(name='vendor_metric')),
        semantics_resolver=resolver,
    )

    assert report.primary_metric is not None
    assert report.primary_metric.semantics.semantic_id == 'quality.accuracy.ratio'


def test_num_counts_one_metric_even_without_a_resolved_primary() -> None:
    """A report whose primary metric cannot be resolved still reports its real sample count.

    Counting one metric is what avoids double-counting a shared sample set; that holds for any
    single metric, so dropping the primary must not turn the sample count into zero.
    """
    report = _report()
    assert report.num == 10

    report.primary_metric_identity = None
    assert report.primary_metric is None
    assert report.num == 10


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


def test_legacy_humaneval_report_recovers_structured_primary_from_metadata() -> None:
    report = Report.from_dict({
        'dataset_name': 'humaneval',
        'metrics': [
            {
                'name': 'mean_acc',
                'score': 0.5,
                'categories': [],
            },
            {
                'name': 'mean_acc_pass@1',
                'score': 0.75,
                'categories': [],
            },
        ],
    })

    assert report.primary_metric_identity == MetricIdentity(
        name='accuracy', aggregation='pass_at_k', dimensions={'k': 1}
    )


def test_legacy_general_qa_report_recovers_rouge_primary_from_metadata() -> None:
    report = Report.from_dict({
        'dataset_name': 'general_qa',
        'metrics': [
            {
                'name': 'Rouge-1-R',
                'score': 0.7,
                'categories': [],
            },
            {
                'name': 'Rouge-L-R',
                'score': 0.8,
                'categories': [],
            },
        ],
    })

    assert report.primary_metric_identity == MetricIdentity(
        name='rouge', aggregation='mean', dimensions={
            'statistic': 'recall',
            'variant': 'l'
        }
    )


def test_agg_score_keeps_deprecated_aggregation_name_compatibility() -> None:
    with pytest.warns(DeprecationWarning, match='aggregation_name'):
        score = AggScore(score=1.0, metric_name='accuracy', aggregation_name='mean')
    with pytest.warns(DeprecationWarning, match='aggregation_name'):
        assert score.aggregation_name == 'mean'

    assert AggScore(score=1.0, metric_name='accuracy').aggregation == 'identity'


@pytest.mark.parametrize(
    ('metric_name', 'expected'),
    [
        ('MyCustomMetric', MetricIdentity(name='my_custom_metric', aggregation='mean')),
        ('weird.metric', MetricIdentity(name='weird_metric', aggregation='mean')),
        ('Third Party Score', MetricIdentity(name='third_party_score', aggregation='mean')),
    ],
)
def test_agg_score_degrades_unknown_non_canonical_names(metric_name: str, expected: MetricIdentity) -> None:
    """A third-party metric name must be usable, not fatal.

    Raising here would abort the whole run for any adapter that spells a metric in CamelCase, and
    would make the resolver's diagnostic degradation unreachable for exactly those metrics.
    """
    score = AggScore(score=0.5, metric_name=metric_name, aggregation='mean')

    assert score.identity == expected


@pytest.mark.parametrize('metric_name', ['', '123', '!!!'])
def test_agg_score_keeps_unnormalizable_names_reportable(metric_name: str) -> None:
    """Nothing snake-caseable left: the value stays reportable and keeps its original spelling."""
    score = AggScore(score=0.5, metric_name=metric_name, aggregation='mean')

    assert score.identity == MetricIdentity(
        name='legacy_metric', aggregation='mean', dimensions={'original_name': metric_name}
    )


@pytest.mark.parametrize('metric_name', ['score', 'overall', 'total_score'])
def test_agg_score_isolates_forbidden_ambiguous_names(metric_name: str) -> None:
    score = AggScore(score=1.0, metric_name=metric_name, aggregation='mean')

    assert score.identity == MetricIdentity(
        name='legacy_metric', aggregation='mean', dimensions={'original_name': metric_name}
    )


@pytest.mark.parametrize('metric_name', ['gpt_score', 'avg_score'])
def test_agg_score_does_not_reinterpret_valid_ambiguous_names(metric_name: str) -> None:
    score = AggScore(score=1.0, metric_name=metric_name, aggregation='mean')

    assert score.identity == MetricIdentity(name=metric_name, aggregation='mean')


def test_agg_score_only_normalizes_overlap_metric_syntax() -> None:
    bleu = AggScore(score=0.5, metric_name='bleu-4', aggregation='mean')
    rouge = AggScore(score=0.5, metric_name='Rouge-L-R', aggregation='mean')

    assert bleu.identity == MetricIdentity(name='bleu_4', aggregation='mean')
    assert rouge.identity == MetricIdentity(name='rouge_l_r', aggregation='mean')
