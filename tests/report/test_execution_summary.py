from evalscope.report import ExecutionSubset, ExecutionSummary, Report
from evalscope.service.blueprints.eval import _all_results_empty


def test_no_score_report_is_persisted_as_incomplete_and_service_failure() -> None:
    report = Report(
        execution_summary=ExecutionSummary(
            requested=2,
            succeeded=0,
            errored=2,
            incomplete=True,
            subsets={'test': ExecutionSubset(requested=2, succeeded=0, errored=2)},
        )
    )

    assert report.score is None
    assert report.execution_summary.incomplete
    assert _all_results_empty(report)
