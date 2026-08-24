import pandas as pd
import pytest
from pydantic import ValidationError
from typing import Dict, List, Optional

from evalscope.api.judge import summarize_judge_runs
from evalscope.api.metric import AggScore, JudgeSummary, SampleScore, Score
from evalscope.api.metric.semantics import MetricIdentity, MetricKind, MetricSelector
from evalscope.constants import ScoreStatus
from evalscope.metrics.semantics.catalog import LEGACY_METRIC_MIGRATIONS
from evalscope.metrics.semantics.entry import MetricEntry
from evalscope.metrics.semantics.migration import migrate_legacy_report_identity
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
    assert all(metric.semantics.kind is MetricKind.QUALITY for metric in report.metrics)


def test_generator_preserves_scores() -> None:
    report = _report()
    assert [metric.score for metric in report.metrics] == [0.8, 0.75]


def test_collection_report_keeps_mixed_metrics_without_primary() -> None:
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
    assert all(metric.semantics.kind is MetricKind.QUALITY for metric in report.metrics)
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
    # A report with no metric produced no score; it did not score zero.
    assert Report().score is None
    assert 'score' not in report.to_dict()


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


def test_report_persists_first_class_judge_summary() -> None:
    report = _report()
    report.judge_summary = JudgeSummary(status=ScoreStatus.DEGRADED, scored=8, total=10, coverage=0.8)

    assert report.to_dict()['judge_summary']['coverage'] == 0.8


def test_run_judge_summary_keeps_unavailable_samples_out_of_scores() -> None:
    usable = SampleScore(score=Score(judge_summary=JudgeSummary(
        status=ScoreStatus.SUCCESS,
        scored=1,
        total=1,
        coverage=1.0,
        judge_models=['primary'],
        valid_observations=1,
        total_observations=1,
    )))
    unavailable = SampleScore(score=Score(judge_summary=JudgeSummary(
        status=ScoreStatus.EXCLUDED,
        scored=0,
        total=1,
        coverage=0.0,
        judge_models=['primary'],
        total_observations=1,
        failures={'parse_error': 1},
    )))

    summary = summarize_judge_runs([[usable, unavailable]])

    assert summary.status is ScoreStatus.DEGRADED
    assert (summary.scored, summary.total, summary.coverage) == (1, 2, 0.5)
    assert summary.failures == {'parse_error': 1}


def test_run_judge_summary_preserves_degradation_and_rolls_up_disagreement() -> None:
    degraded = SampleScore(score=Score(judge_summary=JudgeSummary(
        status=ScoreStatus.DEGRADED,
        scored=1,
        total=1,
        coverage=1.0,
        disagreement={
            'numeric': {'all_observations': {'acc': {'std': 0.2, 'range': 0.5}}},
            'categorical': {'pair': {'agreement_ratio': 0.5, 'vote_entropy': 1.0}},
            'position_consistency': 0.5,
            'swap_flip_count': 1,
        },
    )))

    summary = summarize_judge_runs([[degraded]])

    assert summary.status is ScoreStatus.DEGRADED
    assert summary.disagreement['numeric']['acc'] == {'mean_std': 0.2, 'max_range': 0.5, 'samples': 1}
    assert summary.disagreement['position_consistency']['swap_flip_count'] == 1


def test_v2_round_trip_uses_persisted_semantics_without_resolution() -> None:
    original = _report()
    restored = Report.from_dict(original.to_dict())
    assert restored == original


def test_persisted_v2_role_contract_migrates_on_read_only() -> None:
    data = _report().to_dict()
    data['primary_metric_identity'] = None
    for metric in data['metrics']:
        semantics = metric['semantics']
        semantics['role'] = 'primary' if metric['identity']['name'] == 'f1' else 'auxiliary'
        semantics['contract_version'] = 1
        semantics.pop('kind')

    report = Report.from_dict(data)

    assert report.primary_metric_identity == MetricIdentity(name='f1', aggregation='mean')
    assert all(metric.semantics.kind is MetricKind.QUALITY for metric in report.metrics)
    for metric in report.to_dict()['metrics']:
        assert 'role' not in metric['semantics']
        assert 'contract_version' not in metric['semantics']


def test_transitional_v1_fields_migrate_to_current_report_shape() -> None:
    report = Report.from_dict({
        'dataset_name': 'general_mcq',
        'metric_schema_version': 1,
        'primary_metric_name': 'mean_acc',
        'metrics': [{
            'name': 'mean_acc',
            'semantic_id': 'quality.accuracy.ratio',
            'score': 0.8,
            'categories': [],
        }],
    })

    assert report.primary_metric_identity == MetricIdentity(name='accuracy', aggregation='mean')
    metric = report.metrics[0]
    assert metric.semantics.semantic_id == 'quality.accuracy.ratio'
    assert 'semantic_id' not in report.to_dict()['metrics'][0]
    assert 'metric_schema_version' not in report.to_dict()
    assert 'primary_metric_name' not in report.to_dict()


def test_v1_report_migrates_without_changing_values() -> None:
    report = Report.from_dict({
        'dataset_name': 'conll2003',
        'score': 0.8,
        'metrics': [{
            'name': 'mean_f1_score',
            'score': 0.8,
            'categories': [],
        }],
    })
    assert report.metrics[0].score == 0.8
    assert report.metrics[0].identity == MetricIdentity(name='f1', aggregation='mean')
    assert report.metrics[0].semantics.kind is MetricKind.QUALITY
    assert report.primary_metric_identity == report.metrics[0].identity
    assert 'name' not in report.to_dict()['metrics'][0]


@pytest.mark.parametrize(('legacy_name', 'entry'), sorted(LEGACY_METRIC_MIGRATIONS.items()))
def test_every_legacy_manifest_entry_keeps_its_declared_semantics(legacy_name: str, entry: MetricEntry) -> None:
    report = Report.from_dict({
        'dataset_name': 'legacy_manifest_test',
        'metrics': [{
            'name': legacy_name,
            'score': 0.73,
            'categories': [],
        }],
    })

    metric = report.metrics[0]
    expected = entry.resolve(metric.identity.name)
    assert metric.identity == migrate_legacy_report_identity(legacy_name, 'legacy_manifest_test')
    assert metric.semantics == expected
    expected_primary = None if expected.kind is MetricKind.DIAGNOSTIC else metric.identity
    assert report.primary_metric_identity == expected_primary


def test_legacy_vqa_score_keeps_unbounded_model_score_semantics() -> None:
    report = Report.from_dict({
        'dataset_name': 'genai_bench',
        'metrics': [{
            'name': 'VQAScore',
            'score': 0.73,
            'categories': [],
        }],
    })

    metric = report.metrics[0]
    assert metric.identity == MetricIdentity(name='vqa_model_score', aggregation='identity')
    assert metric.score == 0.73
    assert metric.semantics.semantic_id == 'quality.model_score.unbounded'
    assert metric.semantics.value_range is None
    assert metric.semantics.kind is MetricKind.QUALITY
    assert report.primary_metric_identity == metric.identity


def test_legacy_and_v2_error_rate_keep_distinct_semantics() -> None:
    legacy_report = Report.from_dict({
        'dataset_name': 'legacy_parser',
        'metrics': [{
            'name': 'error_rate',
            'score': 0.2,
            'categories': [],
        }],
    })
    current_report = ReportGenerator.generate_report(
        {'test': [AggScore(score=0.2, metric_name='error_rate', aggregation='mean', num=1)]},
        'model',
        _StubAdapter('current_benchmark'),
    )

    assert legacy_report.metrics[0].semantics.semantic_id == 'diagnostic.parse_status.ratio'
    assert legacy_report.metrics[0].semantics.kind is MetricKind.DIAGNOSTIC
    assert current_report.metrics[0].semantics.semantic_id == 'quality.error_rate.ratio'
    assert current_report.metrics[0].semantics.kind is MetricKind.QUALITY
    assert current_report.primary_metric_identity == current_report.metrics[0].identity


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
    assert metric.semantics.kind is MetricKind.DIAGNOSTIC
    assert report.primary_metric is None


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
    assert metric.semantics.kind is MetricKind.DIAGNOSTIC


def test_unknown_valid_third_party_identity_can_be_written_as_diagnostic() -> None:
    report = ReportGenerator.generate_report(
        {'test': [AggScore(score=3.5, metric_name='vendor_metric', aggregation='mean', num=1)]},
        'model',
        _StubAdapter('third_party'),
    )
    assert report.metrics[0].identity.name == 'vendor_metric'
    assert report.metrics[0].semantics.kind is MetricKind.DIAGNOSTIC
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


@pytest.mark.parametrize(
    ('metric_name', 'canonical_name'),
    [
        ('acc', 'accuracy'),
        ('ACC', 'accuracy'),
        ('bertscore', 'bert_score'),
        ('f1_score', 'f1'),
        ('F1', 'f1'),
        ('em', 'exact_match'),
    ],
)
def test_agg_score_accepts_only_safe_producer_aliases(metric_name: str, canonical_name: str) -> None:
    score = AggScore(score=0.5, metric_name=metric_name, aggregation='mean')

    assert score.identity == MetricIdentity(name=canonical_name, aggregation='mean')


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


@pytest.mark.parametrize(
    'fields',
    [
        {
            'aggregation': '!!!'
        },
        {
            'aggregation': 'mean', 'dimensions': {
                'Invalid Key': 1
            }
        },
    ],
)
def test_agg_score_rejects_invalid_explicit_structure(fields) -> None:
    with pytest.raises(ValidationError):
        AggScore(score=1.0, metric_name='accuracy', **fields)


def test_agg_score_structures_overlap_metric_syntax_for_primary_selection() -> None:
    bleu = AggScore(score=0.5, metric_name='bleu-4', aggregation='mean')
    rouge = AggScore(score=0.5, metric_name='Rouge-L-R', aggregation='mean')

    assert bleu.identity == MetricIdentity(name='bleu', aggregation='mean', dimensions={'ngram': 4})
    assert rouge.identity == MetricIdentity(
        name='rouge', aggregation='mean', dimensions={'statistic': 'recall', 'variant': 'l'}
    )

    report = ReportGenerator.generate_report(
        {'test': [bleu, rouge]},
        'model',
        _StubAdapter(
            'general_qa',
            MetricSelector(name='rouge', aggregation='mean', dimensions={'statistic': 'recall', 'variant': 'l'}),
        ),
    )

    assert report.primary_metric_identity == rouge.identity
