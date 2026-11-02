import pytest

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


@pytest.mark.parametrize(
    'path, query, expected',
    [
        ('table', {'view': 'dataset', 'dataset_name': 'terminal_bench_v2'}, 'columns'),
        ('charts/sunburst', {}, 'No data to plot'),
        ('charts/dataset_scores', {'dataset_name': 'terminal_bench_v2'}, 'No data to plot'),
    ],
)
def test_no_score_report_renders_as_empty_in_the_service(tmp_path, path: str, query: dict, expected: str) -> None:
    pytest.importorskip('flask')
    from evalscope.service.app import create_app

    # All samples failed under ignore_errors: the evaluator still writes this report for the dashboard to open.
    _no_score_report().to_json(str(tmp_path / 'run' / 'reports' / 'test-model' / 'terminal_bench_v2.json'))
    (tmp_path / 'run' / 'configs').mkdir()
    (tmp_path / 'run' / 'configs' / 'task_config.yaml').write_text('model: test-model\n', encoding='utf-8')

    response = create_app().test_client().get(
        f'/api/v1/reports/runs/run/models/test-model/{path}',
        query_string={'root_path': str(tmp_path), **query},
    )

    assert response.status_code == 200
    assert expected in response.get_data(as_text=True)
