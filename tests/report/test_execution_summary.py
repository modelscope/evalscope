from evalscope.report import ExecutionSubset, ExecutionSummary, Report, get_display_data_frame, get_report_list
from evalscope.service.blueprints.eval import _all_results_empty


def _no_score_report() -> Report:
    return Report(
        name='terminal_bench_v2',
        dataset_name='terminal_bench_v2',
        model_name='test-model',
        execution_summary=ExecutionSummary(
            requested=2,
            succeeded=0,
            errored=2,
            incomplete=True,
            subsets={'test': ExecutionSubset(requested=2, succeeded=0, errored=2)},
        ),
    )


def test_no_score_report_is_persisted_as_incomplete_and_service_failure() -> None:
    report = _no_score_report()

    assert report.score is None
    assert report.execution_summary.incomplete
    assert _all_results_empty(report)
    assert _all_results_empty({'terminal_bench_v2': report})


def test_no_score_report_is_readable_and_renders_as_an_empty_table(tmp_path) -> None:
    report_file = tmp_path / 'reports' / 'terminal_bench_v2.json'
    _no_score_report().to_json(str(report_file))

    reports = get_report_list([str(report_file.parent)])

    assert len(reports) == 1
    assert reports[0].execution_summary.incomplete
    assert get_display_data_frame(reports).empty
