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
from evalscope.api.metric.semantics import MetricRole
from evalscope.metrics.semantics import format_metric_value
from evalscope.report.report import Report
from evalscope.utils.data_utils import get_acc_report_df


class _StubAdapter:
    """Minimal adapter surface the report generator reads."""

    def __init__(self, name: str, primary_metric: Optional[str] = None, aggregation: str = 'mean') -> None:
        self.name = name
        self.primary_metric = primary_metric
        self.aggregation = aggregation
        self.pretty_name = name
        self.description = ''
        self.category_map: Dict[str, List[str]] = {}


#: (scenario, benchmark, primary raw name, aggregated scores, expected semantic id of the primary)
SCENARIOS = [
    (
        'accuracy',
        'gsm8k',
        None,
        [AggScore(score=0.8567, metric_name='acc', aggregation_name='mean', num=100)],
        'quality.accuracy.ratio',
    ),
    (
        'wer',
        'torgo',
        'wer',
        [
            AggScore(score=0.0432, metric_name='wer', aggregation_name='', num=50),
            AggScore(score=0.0321, metric_name='cer', aggregation_name='', num=50),
        ],
        'quality.wer.ratio',
    ),
    (
        'pass_at_1',
        'humaneval',
        None,
        [AggScore(score=0.75, metric_name='acc', aggregation_name='mean', num=164)],
        'quality.accuracy.ratio',
    ),
    (
        'ner_f1',
        'conll2003',
        'f1_score',
        [
            AggScore(score=0.91, metric_name='f1_score', aggregation_name='', num=200),
            AggScore(score=0.89, metric_name='precision', aggregation_name='', num=200),
            AggScore(score=0.93, metric_name='recall', aggregation_name='', num=200),
        ],
        'quality.f1.ratio',
    ),
    (
        'official_points_100',
        'arena_hard',
        None,
        [AggScore(score=87.25, metric_name='WeightedScorePercent', aggregation_name='', num=500)],
        'quality.score.points_100',
    ),
    (
        'agent_diagnostics',
        'miniwob',
        'success_rate',
        [
            AggScore(score=0.62, metric_name='success_rate', aggregation_name='mean', num=80),
            AggScore(score=0.11, metric_name='error_rate', aggregation_name='mean', num=80),
        ],
        'quality.accuracy.ratio',
    ),
]


def _build_report(benchmark: str, primary_metric: Optional[str], agg_scores: List[AggScore]) -> Report:
    """Generate a report the way the evaluator does, then round-trip it through JSON."""
    from evalscope.report.generator import ReportGenerator

    report = ReportGenerator.generate_report(
        score_dict={'default': agg_scores},
        model_name='test-model',
        data_adapter=_StubAdapter(benchmark, primary_metric=primary_metric),
        add_aggregation_name=True,
    )
    # Reading a report back is the path every surface takes, so assert on that shape.
    return Report.from_dict(json.loads(json.dumps(report.to_dict())))


@pytest.mark.parametrize('scenario,benchmark,primary_metric,agg_scores,expected_semantic_id', SCENARIOS)
class TestSemanticsEndToEnd:
    """Feature: metric-semantics-governance, Property 41: every surface agrees."""

    def test_primary_metric_resolves_to_the_expected_semantics(
        self, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id
    ) -> None:
        report = _build_report(benchmark, primary_metric, agg_scores)

        assert report.primary_metric is not None, scenario
        assert report.primary_metric.semantics.semantic_id == expected_semantic_id
        assert report.primary_metric.semantics.role is MetricRole.PRIMARY

    def test_exactly_one_primary_metric(
        self, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id
    ) -> None:
        report = _build_report(benchmark, primary_metric, agg_scores)

        primaries = [m for m in report.metrics if m.semantics and m.semantics.role is MetricRole.PRIMARY]
        assert len(primaries) == 1, f'{scenario}: {[m.name for m in primaries]}'

    def test_cli_dataframe_matches_the_primary_metric(
        self, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id
    ) -> None:
        report = _build_report(benchmark, primary_metric, agg_scores)
        df, _ = get_acc_report_df([report])

        assert len(df) == 1
        assert df.iloc[0]['Score'] == pytest.approx(report.primary_metric.score)

    def test_html_report_shows_the_same_formatted_value(
        self, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id, tmp_path: Path
    ) -> None:
        from evalscope.report.renderer import gen_html_report_file

        reports_dir = tmp_path / 'reports' / 'test-model'
        reports_dir.mkdir(parents=True)
        report = _build_report(benchmark, primary_metric, agg_scores)
        (reports_dir / f'{benchmark}.json').write_text(report.to_json_str(), encoding='utf-8')

        html_path = gen_html_report_file(str(tmp_path / 'reports'))
        html = Path(html_path).read_text(encoding='utf-8')

        semantics = report.primary_metric.semantics
        expected_value = format_metric_value(report.primary_metric.score, semantics)
        assert semantics.metric_name in html, scenario
        assert expected_value in html, f'{scenario}: expected {expected_value!r} in the HTML report'

    def test_api_payload_carries_the_same_contract(
        self, scenario, benchmark, primary_metric, agg_scores, expected_semantic_id
    ) -> None:
        from evalscope.service.blueprints.reports import _report_to_service_dict

        report = _build_report(benchmark, primary_metric, agg_scores)
        payload = _report_to_service_dict(report)

        assert payload['primary_metric_name'] == report.primary_metric.name
        by_name = {metric['name']: metric for metric in payload['metrics']}
        primary_payload = by_name[report.primary_metric.name]
        assert primary_payload['semantics']['semantic_id'] == expected_semantic_id
        assert primary_payload['semantics']['role'] == 'primary'
        # The persisted anchor and the hydrated contract must not disagree.
        assert primary_payload['semantic_id'] == primary_payload['semantics']['semantic_id']


class TestDirectionsSurviveTheRoundTrip:
    """A low-is-better metric must not be presented as if higher were better."""

    def test_wer_is_lower_is_better_everywhere(self) -> None:
        report = _build_report('torgo', 'wer', [
            AggScore(score=0.0432, metric_name='wer', aggregation_name='', num=50),
            AggScore(score=0.0321, metric_name='cer', aggregation_name='', num=50),
        ])

        assert report.primary_metric.semantics.direction.value == 'lower_is_better'
        # The supporting error rate stays comparable but is not the conclusion.
        cer = next(metric for metric in report.metrics if metric.name == 'cer')
        assert cer.semantics.role is MetricRole.AUXILIARY
        assert cer.semantics.direction.value == 'lower_is_better'

    def test_diagnostics_never_become_the_conclusion(self) -> None:
        report = _build_report('miniwob', 'success_rate', [
            AggScore(score=0.62, metric_name='success_rate', aggregation_name='mean', num=80),
            # Emitted without an aggregation prefix, as the adapters that report counts do.
            AggScore(score=0.5, metric_name='no_answer_num', aggregation_name='', num=80),
        ])

        diagnostic = next(metric for metric in report.metrics if metric.name == 'no_answer_num')
        assert diagnostic.semantics.role is MetricRole.DIAGNOSTIC
        assert diagnostic.semantics.direction.value == 'none'
        assert diagnostic.semantics.comparison_group is None
        assert report.primary_metric.name == 'mean_success_rate'


class TestPerfSemanticsSurface:
    """The perf API attaches semantics without touching the numbers it reports."""

    def test_public_perf_fields_all_resolve(self) -> None:
        from evalscope.metrics.semantics.perf import resolve_perf_semantics
        from evalscope.perf.utils.perf_constants import Metrics, PercentileMetrics

        keys = [Metrics.AVERAGE_LATENCY, Metrics.OUTPUT_TOKEN_THROUGHPUT, PercentileMetrics.TTFT]
        semantics = resolve_perf_semantics(keys)

        assert set(semantics) == set(keys)
        assert semantics[Metrics.AVERAGE_LATENCY]['direction'] == 'lower_is_better'
        assert semantics[Metrics.OUTPUT_TOKEN_THROUGHPUT]['direction'] == 'higher_is_better'

    def test_counts_are_diagnostics(self) -> None:
        from evalscope.metrics.semantics.perf import resolve_perf_semantics
        from evalscope.perf.utils.perf_constants import Metrics

        semantics = resolve_perf_semantics([Metrics.FAILED_REQUESTS, Metrics.TOTAL_REQUESTS])

        for field_key in (Metrics.FAILED_REQUESTS, Metrics.TOTAL_REQUESTS):
            assert semantics[field_key]['role'] == 'diagnostic'
            assert semantics[field_key]['direction'] == 'none'
