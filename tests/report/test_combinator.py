import json
from typing import List, Optional

from evalscope.api.metric import AggScore
from evalscope.api.metric.semantics import MetricIdentity, MetricSelector
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.metrics.semantics.identity import migrate_legacy_identity
from evalscope.report import gen_table, get_report_list
from evalscope.report.generator import ReportGenerator
from evalscope.report.report import Category, Metric, Report, Subset


class _StubAdapter:

    def __init__(self, name: str, primary_metric: Optional[str] = None, aggregation: str = 'mean') -> None:
        self.name = name
        self.primary_metric = MetricSelector(
            name=migrate_legacy_identity(primary_metric, aggregation).name
        ) if primary_metric else None
        self.aggregation = aggregation
        self.pretty_name = name
        self.description = ''
        self.category_map = {}


def _report(benchmark: str, scores: List[AggScore], primary_metric: Optional[str] = None) -> Report:
    return ReportGenerator.generate_report(
        score_dict={'default': scores},
        model_name='test-model',
        data_adapter=_StubAdapter(benchmark, primary_metric=primary_metric),
    )


def _categorized_report(category_name: tuple[str, ...]) -> Report:
    report = _report('gsm8k', [AggScore(score=0.8, metric_name='accuracy', aggregation='mean', num=5)])
    report.metrics[0].categories = [Category(name=category_name, subsets=[Subset(name='main', score=0.8, num=5)])]
    return report


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
                identity=MetricIdentity(name='submission_ready', aggregation='mean'),
                semantics=get_semantics_resolver().resolve(
                    'gdpval', MetricIdentity(name='submission_ready', aggregation='mean')
                ).semantics,
                categories=[Category(
                    name=('default', ),
                    subsets=[Subset(name='default', score=0.8679, num=1)],
                )],
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


def test_gen_table_formats_metric_values_without_changing_dataframe_scores() -> None:
    accuracy = _report('gsm8k', [AggScore(score=0.8567, metric_name='accuracy', aggregation='mean', num=100)])
    accuracy.dataset_pretty_name = 'GSM8K Pretty'
    wer = _report(
        'torgo', [AggScore(score=0.0432, metric_name='wer', aggregation='identity', num=50)], primary_metric='wer'
    )
    cider = _report('cider', [AggScore(score=1.23456, metric_name='cider', aggregation='mean', num=10)])
    diagnostic = _report(
        'third_party', [AggScore(score=0.87654321, metric_name='mystery', aggregation='identity', num=2)]
    )

    table = gen_table(report_list=[accuracy, wer, cider, diagnostic])

    assert 'Accuracy ↑' in table
    assert 'GSM8K Pretty' in table
    assert '85.7%' in table
    assert 'WER ↓' in table
    assert '4.3%' in table
    assert 'CIDEr ↑' in table
    assert '1.235' in table
    assert 'mystery' in table
    assert '0.8765' in table
    assert '0.8567' not in table
    assert accuracy.to_dataframe()['Score'].tolist() == [0.8567]


def test_gen_table_disambiguates_repeated_metric_display_names() -> None:
    report = _report(
        'plawbench',
        [
            AggScore(score=0.72, metric_name='accuracy', aggregation='mean', num=30),
            AggScore(score=0.75, metric_name='fact_acc', aggregation='mean', num=30),
        ],
        primary_metric='accuracy',
    )

    table = gen_table(report_list=[report])

    assert 'Accuracy ↑ (accuracy:mean)' in table
    assert 'Accuracy ↑ (fact_acc:mean)' in table


def test_gen_table_hides_placeholder_category_columns_without_changing_dataframe() -> None:
    report = _categorized_report(('default', ))
    raw_dataframe = report.to_dataframe()

    table = gen_table(report_list=[report])

    assert 'Cat.0' not in table
    assert 'Category' not in table
    assert 'default' not in table
    assert raw_dataframe['Cat.0'].tolist() == ['default']


def test_gen_table_renames_one_informative_category_level() -> None:
    table = gen_table(report_list=[_categorized_report(('math', ))])

    assert 'Category' in table
    assert 'Cat.0' not in table
    assert 'math' in table


def test_gen_table_numbers_multiple_informative_category_levels() -> None:
    table = gen_table(report_list=[_categorized_report(('knowledge', 'math'))])

    assert 'Category 1' in table
    assert 'Category 2' in table
    assert 'Cat.0' not in table
    assert 'Cat.1' not in table
    assert 'knowledge' in table
    assert 'math' in table


def test_gen_table_replaces_placeholder_categories_when_mixed_with_informative_values() -> None:
    default_report = _categorized_report(('default', ))
    categorized_report = _categorized_report(('math', ))
    categorized_report.dataset_name = 'categorized'

    table = gen_table(report_list=[default_report, categorized_report])

    assert 'Category' in table
    assert 'math' in table
    assert 'default' not in table
    assert default_report.to_dataframe()['Cat.0'].tolist() == ['default']
