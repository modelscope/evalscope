import json

from evalscope.report import gen_table, get_report_list
from evalscope.report.report import Category, Metric, Report, Subset


def test_get_report_list_skips_non_report_json(tmp_path):
    reports_dir = tmp_path / 'reports'
    report_file = reports_dir / 'qwen-plus' / 'gdpval.json'
    submission_info_file = reports_dir / 'qwen-plus' / 'gdpval_submission' / 'submission_info.json'
    empty_report_like_file = reports_dir / 'qwen-plus' / 'empty_report_like.json'

    report = Report(
        name='qwen-plus@gdpval',
        dataset_name='gdpval',
        model_name='qwen-plus',
        metrics=[
            Metric(
                name='mean_llm_rubric_score',
                categories=[
                    Category(
                        name=('default', ),
                        subsets=[Subset(name='default', score=0.8679, num=1)],
                    )
                ],
            )
        ],
    )
    report.to_json(str(report_file))

    submission_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(submission_info_file, 'w', encoding='utf-8') as f:
        json.dump({'benchmark': 'gdpval', 'samples': [{'id': 'case-1'}]}, f)
    with open(empty_report_like_file, 'w', encoding='utf-8') as f:
        json.dump({'name': 'empty', 'dataset_name': 'empty', 'model_name': 'qwen-plus', 'metrics': []}, f)

    reports = get_report_list([str(reports_dir)])
    assert len(reports) == 1
    assert reports[0].dataset_name == 'gdpval'
    assert reports[0].model_name == 'qwen-plus'

    table = gen_table(reports_path_list=[str(reports_dir)], add_overall_metric=True)
    assert 'gdpval' in table
    assert 'qwen-plus' in table
    assert 'default_dataset' not in table
