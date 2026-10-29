"""Primary metric selection, the one policy shared by every report producer.

Selection reports objective outcomes. Report producers decide whether an outcome requires an
error, a compatibility fallback, or an unavailable-primary explanation.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from evalscope.api.metric.semantics import MetricIdentity, MetricKind, MetricSelector, MetricSemantics
from evalscope.metrics.semantics.migration import hydrate_report_semantics
from evalscope.metrics.semantics.primary import PrimarySelectionStatus, read_meta_primary_selector, select_primary
from evalscope.metrics.semantics.resolver import get_semantics_resolver
from evalscope.report.report import Report


def _semantics(identities: List[MetricIdentity], benchmark_name: str = 'benchmark') -> Dict[str, MetricSemantics]:
    resolver = get_semantics_resolver()
    return {identity.key: resolver.resolve(benchmark_name, identity).semantics for identity in identities}


def test_structured_selector_selects_one_identity_without_mutating_semantics() -> None:
    identities = [
        MetricIdentity(name='rouge', aggregation='mean', dimensions={
            'ngram': 1,
            'statistic': 'recall'
        }),
        MetricIdentity(name='rouge', aggregation='mean', dimensions={
            'statistic': 'recall',
            'variant': 'l'
        }),
    ]
    semantics = _semantics(identities, 'general_qa')
    selector = MetricSelector(name='rouge', aggregation='mean', dimensions={'variant': 'l', 'statistic': 'recall'})

    selection = select_primary(identities, semantics, selector)

    assert selection.identity == identities[1]
    assert selection.unavailable_reason is None
    assert selection.status is PrimarySelectionStatus.SELECTED
    assert all(item.kind is MetricKind.QUALITY for item in semantics.values())


def test_only_one_quality_identity_can_be_implicit_primary() -> None:
    identity = MetricIdentity(name='accuracy', aggregation='mean')

    selection = select_primary([identity], _semantics([identity]), None)

    assert selection.identity == identity


class TestMissingPrimary:
    """A selector that matches nothing describes the run, so a report may still be written."""

    def test_zero_matches_reports_a_reason_without_blaming_the_author(self) -> None:
        identities = [MetricIdentity(name='accuracy', aggregation='mean')]

        selection = select_primary(identities, _semantics(identities), MetricSelector(name='recall'))

        assert selection.identity is None
        assert selection.status is PrimarySelectionStatus.NO_MATCH
        assert 'did not match' in selection.unavailable_reason

    def test_a_report_of_only_diagnostics_has_no_conclusion(self) -> None:
        identities = [MetricIdentity(name='no_answer_num', aggregation='mean')]

        selection = select_primary(identities, _semantics(identities), None)

        assert selection.identity is None
        assert selection.status is PrimarySelectionStatus.NO_SCORED_METRICS


class TestNonSelectablePrimary:
    """Ambiguity and diagnostic matches are reported without choosing a primary."""

    def test_selector_matching_several_identities(self) -> None:
        identities = [
            MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'a'}),
            MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'b'}),
        ]

        selection = select_primary(identities, _semantics(identities), MetricSelector(name='accuracy'))

        assert selection.identity is None
        assert selection.status is PrimarySelectionStatus.AMBIGUOUS
        assert 'matched 2 identities' in selection.unavailable_reason

    def test_selector_matching_a_diagnostic(self) -> None:
        identities = [MetricIdentity(name='no_answer_num', aggregation='mean')]

        selection = select_primary(identities, _semantics(identities), MetricSelector(name='no_answer_num'))

        assert selection.identity is None
        assert selection.status is PrimarySelectionStatus.DIAGNOSTIC

    def test_several_scored_metrics_without_a_selector(self) -> None:
        identities = [
            MetricIdentity(name='accuracy', aggregation='mean'),
            MetricIdentity(name='f1', aggregation='mean'),
        ]

        selection = select_primary(identities, _semantics(identities), None)

        assert selection.identity is None
        assert selection.status is PrimarySelectionStatus.AMBIGUOUS
        assert '2 non-diagnostic metric identities' in selection.unavailable_reason
        assert 'BenchmarkMeta' not in selection.unavailable_reason


class TestReadingTheDeclaredSelector:
    def test_a_string_declaration_is_migrated_to_a_canonical_selector(self, tmp_path: Path) -> None:
        (tmp_path / 'probe.json').write_text(
            json.dumps({'meta': {'primary_metric': 'mean_acc', 'aggregation': 'mean'}}), encoding='utf-8'
        )

        assert read_meta_primary_selector('probe', tmp_path) == MetricSelector(name='accuracy')

    def test_a_structured_declaration_is_used_as_is(self, tmp_path: Path) -> None:
        selector = {'name': 'f1', 'aggregation': 'macro_mean', 'dimensions': {'scope': 'overall'}}
        (tmp_path / 'probe.json').write_text(json.dumps({'meta': {'primary_metric': selector}}), encoding='utf-8')

        assert read_meta_primary_selector('probe', tmp_path) == MetricSelector.model_validate(selector)

    @pytest.mark.parametrize(
        'payload',
        ['not json at all', json.dumps({'meta': {}}), json.dumps({'meta': {'primary_metric': ''}}),
         json.dumps({'meta': {'primary_metric': {'name': 'total_score'}}})],
    )
    def test_an_unusable_declaration_degrades_to_no_selector(self, tmp_path: Path, payload: str) -> None:
        (tmp_path / 'probe.json').write_text(payload, encoding='utf-8')

        assert read_meta_primary_selector('probe', tmp_path) is None

    def test_an_absent_benchmark_has_no_selector(self, tmp_path: Path) -> None:
        assert read_meta_primary_selector('never_declared', tmp_path) is None


def test_a_stored_identity_that_misses_a_constrained_axis_still_yields_a_primary() -> None:
    """A v1 identity can lack a dimension the current declaration constrains.

    Reading such a report falls back to implicit selection, so its single scored metric is still
    named as the conclusion instead of the report losing its score entirely.
    """
    report = Report.from_dict({
        'dataset_name': 'genai_bench',
        'metrics': [{
            'name': 'VQAScore',
            'score': 0.73,
            'categories': [],
        }],
    })

    assert report.primary_metric_identity == MetricIdentity(name='vqa_model_score', aggregation='identity')
    assert report.primary_metric_unavailable_reason is None
    assert report.score == 0.73


@pytest.mark.parametrize(
    ('metric_names', 'selector_name', 'expected_primary'),
    [
        (['accuracy'], 'recall', 'accuracy'),
        (['accuracy', 'f1'], 'recall', None),
        (['accuracy', 'mean_acc'], 'accuracy', None),
        (['accuracy', 'no_answer_num'], 'no_answer_num', None),
        (['accuracy', 'f1'], None, None),
        (['no_answer_num'], None, None),
    ],
)
def test_read_path_falls_back_only_for_an_unmatched_selector(
    metric_names: List[str], selector_name: Optional[str], expected_primary: Optional[str]
) -> None:
    """Historical ambiguity stays readable; a diagnostic match cannot promote an auxiliary score."""
    report = Report.from_dict({
        'dataset_name': 'legacy_status_probe',
        'metrics': [{'name': name, 'score': 0.73, 'categories': []} for name in metric_names],
    })
    selector = MetricSelector(name=selector_name) if selector_name else None

    hydrate_report_semantics(report, selector_for=lambda _: selector)

    assert [metric.score for metric in report.metrics] == [0.73] * len(metric_names)
    if expected_primary:
        assert report.primary_metric_identity.name == expected_primary
        assert report.score == 0.73
        assert report.primary_metric_unavailable_reason is None
    else:
        assert report.primary_metric_identity is None
        assert report.score is None
        assert report.primary_metric_unavailable_reason
        assert 'BenchmarkMeta' not in report.primary_metric_unavailable_reason
    assert Report.from_dict(report.to_dict()) == report
