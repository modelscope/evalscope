"""Tests for the report model's semantics binding.

* ``TestPrimaryMetricByRole`` -- Property 17: the primary metric is decided by role alone.
* ``TestSerializationContract`` -- Property 40: only ``semantic_id`` is persisted and a
  ``to_dict`` -> ``from_dict`` round trip rebuilds the full contract.
* ``TestGeneratorBinding`` -- Property 15 / 16: every metric is bound and no score changes.
* ``TestLegacyReports`` -- Property 13: a report without anchors renders like a fresh one.
"""

import pytest
from typing import Dict, List, Optional

from evalscope.api.metric import AggScore
from evalscope.api.metric.semantics import METRIC_CONTRACT_VERSION, MetricRole
from evalscope.metrics.semantics import format_metric_value
from evalscope.report.generator import ReportGenerator
from evalscope.report.report import Category, Metric, Report, Subset


class _StubAdapter:
    """Minimal adapter surface used by ``ReportGenerator.generate_report``."""

    def __init__(self, name: str, primary_metric: Optional[str] = None, aggregation: str = 'mean') -> None:
        self.name = name
        self.primary_metric = primary_metric
        self.aggregation = aggregation
        self.pretty_name = name
        self.description = ''
        self.category_map: Dict[str, List[str]] = {}


def _metric(name: str, score: float = 0.5, num: int = 4) -> Metric:
    """Build a metric with one subset carrying ``score``."""
    return Metric(
        name=name, categories=[Category(name=('default', ), subsets=[Subset(name='main', score=score, num=num)])]
    )


def _ner_score_dict() -> Dict[str, List[AggScore]]:
    """Scores shaped like a NER benchmark: F1 plus its supporting metrics."""
    return {
        'default': [
            AggScore(score=0.80, metric_name='f1_score', aggregation_name='', num=10),
            AggScore(score=0.75, metric_name='precision', aggregation_name='', num=10),
            AggScore(score=0.85, metric_name='recall', aggregation_name='', num=10),
            AggScore(score=0.90, metric_name='no_answer_num', aggregation_name='', num=10),
        ]
    }


class TestPrimaryMetricByRole:
    """Feature: metric-semantics-governance: the primary metric is the one whose semantics carry
    role=primary, whatever the order or the names. When no metric declares that role, one is
    inferred so the report still shows a value, and the inference is reported as such."""

    def test_overall_name_does_not_win(self) -> None:
        # `overall` is listed second. With no semantics resolved it holds no privileged position,
        # so the inference falls back to order and picks the first metric, not the name.
        report = Report(dataset_name='conll2003', metrics=[_metric('f1_score'), _metric('overall')])

        assert report.primary_metric is not None
        assert report.primary_metric.name == 'f1_score'
        assert report.primary_metric_is_inferred()
        # An inference is never persisted: the resolver reads this field to decide the role, so
        # writing a guess into it would let the guess define the semantics.
        assert report.primary_metric_name is None

    def test_primary_is_selected_by_role(self) -> None:
        report = Report.from_dict({
            'dataset_name': 'conll2003',
            'metrics': [
                {
                    'name': 'precision',
                    'categories': [],
                    'semantic_id': 'quality.precision.ratio'
                },
                {
                    'name': 'f1_score',
                    'categories': [],
                    'semantic_id': 'quality.f1.ratio'
                },
            ],
        })

        assert report.primary_metric is not None
        assert report.primary_metric.name == 'f1_score'
        assert report.primary_metric.semantics.role is MetricRole.PRIMARY
        assert not report.primary_metric_is_inferred()

    def test_undeclared_metric_still_yields_a_headline_marked_as_inferred(self) -> None:
        # A third-party metric degrades to diagnostic, which used to leave the report with no
        # headline at all. Showing its only number is more useful than showing nothing, as long as
        # the report says the choice was inferred rather than declared.
        report = Report.from_dict({
            'dataset_name': 'unknown_third_party_benchmark',
            'metrics': [{
                'name': 'mystery_metric',
                'categories': []
            }],
        })

        assert report.primary_metric is not None
        assert report.primary_metric.name == 'mystery_metric'
        assert report.primary_metric_is_inferred()
        assert report.primary_metric_name is None

    def test_a_graded_metric_is_preferred_over_a_diagnostic_when_inferring(self) -> None:
        report = Report.from_dict({
            'dataset_name': 'unknown_third_party_benchmark',
            'metrics': [
                {
                    'name': 'no_answer_num',
                    'categories': [],
                    'semantic_id': 'diagnostic.count.items'
                },
                {
                    'name': 'cer',
                    'categories': [],
                    'semantic_id': 'quality.cer.ratio'
                },
            ],
        })

        # The count comes first but describes the run rather than grading it.
        assert report.primary_metric is not None
        assert report.primary_metric.name == 'cer'


class TestSerializationContract:
    """Feature: metric-semantics-governance, Property 40: the report persists the anchor only,
    and hydrating a round trip rebuilds the identical contract."""

    def _generated_report(self) -> Report:
        return ReportGenerator.generate_report(
            score_dict=_ner_score_dict(),
            model_name='m',
            data_adapter=_StubAdapter('conll2003', primary_metric='f1_score'),
            add_aggregation_name=False,
        )

    def test_to_dict_persists_only_the_anchor(self) -> None:
        data = self._generated_report().to_dict()

        for metric_data in data['metrics']:
            assert 'semantic_id' in metric_data
            assert 'semantics' not in metric_data

    def test_schema_version_and_primary_name_are_persisted(self) -> None:
        data = self._generated_report().to_dict()

        assert data['metric_schema_version'] == METRIC_CONTRACT_VERSION
        assert data['primary_metric_name'] == 'f1_score'

    def test_legacy_fields_are_preserved(self) -> None:
        data = self._generated_report().to_dict()

        assert isinstance(data['score'], float)
        assert all('score' in metric_data for metric_data in data['metrics'])

    def test_round_trip_rebuilds_the_contract(self) -> None:
        original = self._generated_report()
        restored = Report.from_dict(original.to_dict())

        assert restored.primary_metric_name == original.primary_metric_name
        assert restored.metric_schema_version == original.metric_schema_version
        for before, after in zip(original.metrics, restored.metrics):
            assert after.semantic_id == before.semantic_id
            assert after.semantics == before.semantics


class TestGeneratorBinding:
    """Feature: metric-semantics-governance, Property 15 and 16: every metric carries semantics
    and no score is altered by binding them."""

    def test_every_metric_is_bound(self) -> None:
        report = ReportGenerator.generate_report(
            score_dict=_ner_score_dict(),
            model_name='m',
            data_adapter=_StubAdapter('conll2003', primary_metric='f1_score'),
            add_aggregation_name=False,
        )

        for metric in report.metrics:
            assert metric.semantics is not None
            assert metric.semantic_id == metric.semantics.semantic_id

    def test_roles_follow_the_declared_primary(self) -> None:
        report = ReportGenerator.generate_report(
            score_dict=_ner_score_dict(),
            model_name='m',
            data_adapter=_StubAdapter('conll2003', primary_metric='f1_score'),
            add_aggregation_name=False,
        )
        roles = {metric.name: metric.semantics.role for metric in report.metrics}

        assert roles['f1_score'] is MetricRole.PRIMARY
        assert roles['precision'] is MetricRole.AUXILIARY
        assert roles['recall'] is MetricRole.AUXILIARY
        assert roles['no_answer_num'] is MetricRole.DIAGNOSTIC

    def test_scores_are_unchanged(self) -> None:
        score_dict = _ner_score_dict()
        expected = {agg.metric_name: agg.score for agg in score_dict['default']}

        report = ReportGenerator.generate_report(
            score_dict=score_dict,
            model_name='m',
            data_adapter=_StubAdapter('conll2003', primary_metric='f1_score'),
            add_aggregation_name=False,
        )

        for metric in report.metrics:
            assert metric.score == pytest.approx(expected[metric.name])

    def test_third_party_undeclared_metric_degrades(self) -> None:
        score_dict = {'default': [AggScore(score=0.5, metric_name='mystery', aggregation_name='', num=2)]}

        report = ReportGenerator.generate_report(
            score_dict=score_dict,
            model_name='m',
            data_adapter=_StubAdapter('a_third_party_benchmark'),
            add_aggregation_name=False,
        )

        assert report.metrics[0].semantics.role is MetricRole.DIAGNOSTIC
        assert report.metrics[0].score == pytest.approx(0.5)

    def test_builtin_undeclared_metric_blocks(self) -> None:
        from evalscope.metrics.semantics import UndeclaredMetricError

        score_dict = {'default': [AggScore(score=0.5, metric_name='not_a_declared_name', aggregation_name='', num=2)]}

        with pytest.raises(UndeclaredMetricError):
            ReportGenerator.generate_report(
                score_dict=score_dict,
                model_name='m',
                data_adapter=_StubAdapter('gsm8k'),
                add_aggregation_name=False,
            )


class TestLegacyReports:
    """Feature: metric-semantics-governance, Property 13: a report written before the contract
    existed hydrates into the same structure as a fresh one, with scores untouched."""

    LEGACY_PAYLOAD = {
        'name': 'm@conll2003',
        'dataset_name': 'conll2003',
        'model_name': 'm',
        'score': 0.8,
        'metrics': [
            {
                'name': 'f1_score',
                'score': 0.8,
                'num': 10,
                'categories': [{
                    'name': ['default'],
                    'subsets': [{
                        'name': 'main',
                        'score': 0.8,
                        'num': 10
                    }],
                }],
            },
            {
                'name': 'precision',
                'score': 0.75,
                'num': 10,
                'categories': [{
                    'name': ['default'],
                    'subsets': [{
                        'name': 'main',
                        'score': 0.75,
                        'num': 10
                    }],
                }],
            },
        ],
    }

    def test_legacy_report_resolves_by_name(self) -> None:
        report = Report.from_dict(dict(self.LEGACY_PAYLOAD))

        assert report.primary_metric is not None
        assert report.primary_metric.name == 'f1_score'
        assert report.metrics[1].semantics.role is MetricRole.AUXILIARY

    def test_legacy_scores_are_untouched(self) -> None:
        report = Report.from_dict(dict(self.LEGACY_PAYLOAD))

        assert report.score == pytest.approx(0.8)
        assert report.metrics[0].score == pytest.approx(0.8)
        assert report.metrics[1].score == pytest.approx(0.75)

    def test_legacy_display_matches_a_fresh_report(self) -> None:
        legacy = Report.from_dict(dict(self.LEGACY_PAYLOAD))
        fresh = ReportGenerator.generate_report(
            score_dict={'main': [AggScore(score=0.8, metric_name='f1_score', aggregation_name='', num=10)]},
            model_name='m',
            data_adapter=_StubAdapter('conll2003'),
            add_aggregation_name=False,
        )

        legacy_semantics = legacy.primary_metric.semantics
        fresh_semantics = fresh.primary_metric.semantics
        assert legacy_semantics.semantic_id == fresh_semantics.semantic_id
        assert legacy_semantics.direction == fresh_semantics.direction
        assert legacy_semantics.display_kind == fresh_semantics.display_kind
        assert format_metric_value(legacy.primary_metric.score, legacy_semantics
                                   ) == format_metric_value(fresh.primary_metric.score, fresh_semantics)

    def test_legacy_report_has_exactly_one_primary(self) -> None:
        report = Report.from_dict(dict(self.LEGACY_PAYLOAD))

        primaries = [m for m in report.metrics if m.semantics and m.semantics.role is MetricRole.PRIMARY]
        assert len(primaries) == 1
