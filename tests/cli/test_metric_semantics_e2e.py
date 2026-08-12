"""End-to-end consistency of metric semantics across every output surface.

A report is generated once per scenario and then read back through each surface that presents it:
the CLI DataFrame, the HTML report, the reports API and the report model itself. The point is not
that each surface works, but that they agree: the same metric must carry the same name, the same
direction and the same formatted value everywhere, because they all read one contract.

No model is called: reports are built from aggregated scores directly, which is what the report
generator consumes.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Optional

from evalscope.api.metric import AggScore
from evalscope.api.metric.semantics import MetricKind, MetricSelector
from evalscope.metrics.semantics import format_metric_value
from evalscope.report.report import Report
from evalscope.utils.data_utils import get_acc_report_df


class _StubAdapter:
    """Minimal adapter surface the report generator reads."""

    def __init__(self, name: str, primary_metric: Optional[MetricSelector] = None) -> None:
        self.name = name
        self.primary_metric = primary_metric
        self.pretty_name = name
        self.description = ''
        self.category_map: Dict[str, List[str]] = {}


#: (scenario, benchmark, primary raw name, aggregated scores, expected semantic id of the primary)
SCENARIOS = [
    (
        'accuracy',
        'gsm8k',
        None,
        [AggScore(score=0.8567, metric_name='accuracy', aggregation='mean', num=100)],
        'quality.accuracy.ratio',
    ),
    (
        'wer',
        'torgo',
        MetricSelector(name='wer', aggregation='identity'),
        [
            AggScore(score=0.0432, metric_name='wer', aggregation='identity', num=50),
            AggScore(score=0.0321, metric_name='cer', aggregation='identity', num=50),
        ],
        'quality.wer.ratio',
    ),
    (
        'pass_at_1',
        'humaneval',
        MetricSelector(name='accuracy', aggregation='pass_at_k', dimensions={'k': 1}),
        [AggScore(score=0.75, metric_name='accuracy', aggregation='pass_at_k', dimensions={'k': 1}, num=164)],
        'quality.pass_at_k.ratio',
    ),
    (
        'ner_f1',
        'conll2003',
        MetricSelector(name='f1', aggregation='identity'),
        [
            AggScore(score=0.91, metric_name='f1', aggregation='identity', num=200),
            AggScore(score=0.89, metric_name='precision', aggregation='identity', num=200),
            AggScore(score=0.93, metric_name='recall', aggregation='identity', num=200),
        ],
        'quality.f1.ratio',
    ),
    (
        'official_points_100',
        'arena_hard',
        None,
        [AggScore(score=87.25, metric_name='weighted_score_percent', aggregation='identity', num=500)],
        'quality.score.points_100',
    ),
    (
        'agent_diagnostics',
        'miniwob',
        MetricSelector(name='success_rate', aggregation='mean'),
        [
            AggScore(score=0.62, metric_name='success_rate', aggregation='mean', num=80),
            AggScore(score=0.11, metric_name='error_rate', aggregation='mean', num=80),
        ],
        'quality.accuracy.ratio',
    ),
    (
        'rubric_dimensions',
        'plawbench',
        MetricSelector(name='accuracy', aggregation='mean'),
        [
            AggScore(score=0.72, metric_name='accuracy', aggregation='mean', num=30),
            AggScore(score=0.68, metric_name='conclusion_acc', aggregation='mean', num=30),
            AggScore(score=0.75, metric_name='fact_acc', aggregation='mean', num=30),
            AggScore(score=0.70, metric_name='reasoning_acc', aggregation='mean', num=30),
            AggScore(score=0.66, metric_name='law_acc', aggregation='mean', num=30),
        ],
        'quality.accuracy.ratio',
    ),
]


def _build_report(benchmark: str, primary_metric: Optional[MetricSelector], agg_scores: List[AggScore]) -> Report:
    """Generate a report the way the evaluator does, then round-trip it through JSON."""
    from evalscope.report.generator import ReportGenerator

    report = ReportGenerator.generate_report(
        score_dict={'default': agg_scores},
        model_name='test-model',
        data_adapter=_StubAdapter(benchmark, primary_metric=primary_metric),
    )
    # Reading a report back is the path every surface takes, so assert on that shape.
    return Report.from_dict(json.loads(json.dumps(report.to_dict())))


@pytest.mark.parametrize('scenario,benchmark,primary_metric,agg_scores,expected_semantic_id', SCENARIOS)
class TestSemanticsEndToEnd:
    """Every metric display surface agrees on the resolved semantics."""

    @pytest.fixture
    def report(self, benchmark, primary_metric, agg_scores) -> Report:
        """The scenario's report, built once per test instead of once per assertion."""
        return _build_report(benchmark, primary_metric, agg_scores)

    def test_primary_metric_resolves_to_the_expected_semantics(
        self, report: Report, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id
    ) -> None:
        assert report.primary_metric is not None, scenario
        assert report.primary_metric.semantics.semantic_id == expected_semantic_id
        assert report.primary_metric.semantics.kind is MetricKind.QUALITY
        assert report.primary_metric.identity == report.primary_metric_identity

    def test_cli_dataframe_matches_the_primary_metric(
        self, report: Report, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id
    ) -> None:
        df = get_acc_report_df([report])

        assert len(df) == 1
        assert df.iloc[0]['Score'] == pytest.approx(report.primary_metric.score)
        assert isinstance(df.iloc[0]['Score'], float)

    def test_html_report_shows_the_same_formatted_value(
        self, report: Report, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id, tmp_path: Path
    ) -> None:
        from evalscope.report.renderer import gen_html_report_file

        reports_dir = tmp_path / 'reports' / 'test-model'
        reports_dir.mkdir(parents=True)
        (reports_dir / f'{benchmark}.json').write_text(report.to_json_str(), encoding='utf-8')

        html_path = gen_html_report_file(str(tmp_path / 'reports'))
        html = Path(html_path).read_text(encoding='utf-8')

        semantics = report.primary_metric.semantics
        expected_value = format_metric_value(report.primary_metric.score, semantics)
        assert semantics.metric_name in html, scenario
        assert expected_value in html, f'{scenario}: expected {expected_value!r} in the HTML report'

    def test_api_payload_carries_the_same_contract(
        self, report: Report, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id
    ) -> None:
        from evalscope.service.blueprints.reports import _report_to_service_dict

        payload = _report_to_service_dict(report)

        assert payload['primary_metric_identity'] == report.primary_metric.identity.model_dump()
        by_identity = {json.dumps(metric['identity'], sort_keys=True): metric for metric in payload['metrics']}
        primary_key = json.dumps(report.primary_metric.identity.model_dump(), sort_keys=True)
        primary_payload = by_identity[primary_key]
        assert primary_payload['semantics']['semantic_id'] == expected_semantic_id
        assert primary_payload['semantics']['kind'] == 'quality'
        assert set(('name', 'semantic_id')).isdisjoint(primary_payload)
        assert set(('score', 'primary_metric_name')).isdisjoint(payload)


class TestDirectionsSurviveTheRoundTrip:
    """A low-is-better metric must not be presented as if higher were better.

    Asserted on the report read back from JSON, which is the shape every surface consumes; the
    per-surface agreement itself is covered by ``TestSemanticsEndToEnd``. The perf key spaces have
    their own resolution tests in ``tests/report/semantics/test_perf_semantics.py``.
    """

    def test_wer_stays_lower_is_better_after_a_round_trip(self) -> None:
        report = _build_report(
            'torgo', MetricSelector(name='wer'), [
                AggScore(score=0.0432, metric_name='wer', aggregation='identity', num=50),
                AggScore(score=0.0321, metric_name='cer', aggregation='identity', num=50),
            ]
        )

        assert report.primary_metric.semantics.direction.value == 'lower_is_better'
        # The supporting error rate stays comparable but is not the conclusion.
        cer = next(metric for metric in report.metrics if metric.identity.name == 'cer')
        assert cer.semantics.kind is MetricKind.QUALITY
        assert cer.semantics.direction.value == 'lower_is_better'

    def test_diagnostics_never_become_the_conclusion(self) -> None:
        report = _build_report(
            'miniwob',
            MetricSelector(name='success_rate'),
            [
                AggScore(score=0.62, metric_name='success_rate', aggregation='mean', num=80),
                # Emitted without an aggregation prefix, as the adapters that report counts do.
                AggScore(score=0.5, metric_name='no_answer_num', aggregation='identity', num=80),
            ]
        )

        diagnostic = next(metric for metric in report.metrics if metric.identity.name == 'no_answer_num')
        assert diagnostic.semantics.kind is MetricKind.DIAGNOSTIC
        assert diagnostic.semantics.direction.value == 'none'
        assert report.primary_metric.identity.name == 'success_rate'
        assert report.primary_metric.identity.aggregation == 'mean'
