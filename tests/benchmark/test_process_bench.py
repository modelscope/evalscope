from pathlib import Path

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.process_bench.process_bench_adapter import ProcessBenchAdapter
from evalscope.config import TaskConfig
from evalscope.report import Report
from evalscope.report.data_frames import get_acc_report_df


def test_comparison_matches_the_published_subset_average(tmp_path: Path) -> None:
    # Rounded subset F1 scores from Table 3 of https://arxiv.org/html/2412.06559v2
    adapter = get_benchmark('process_bench', TaskConfig(datasets=['process_bench']))
    sizes = {'gsm8k': 400, 'math': 1000, 'olympiadbench': 1000, 'omnimath': 1000}
    reports = []
    for model, values in [
        ('Llama-3.1-70B-Instruct', [0.749, 0.482, 0.467, 0.410]),
        ('Qwen2.5-14B-Instruct', [0.693, 0.533, 0.450, 0.413]),
    ]:
        scores = {
            subset: [AggScore(metric_name='simple_f1_score', score=value, num=sizes[subset])]
            for subset, value in zip(sizes, values)
        }
        report = adapter.generate_report(scores, model_name=model, output_dir=str(tmp_path))
        assert report.score == round(sum(values) / len(values), 4)
        subset_scores = {s.name: s.score for c in report.primary_metric.categories for s in c.subsets}
        assert subset_scores == dict(zip(sizes, values))
        path = str(tmp_path / f'{model}.json')
        report.to_json(path)
        reports.append(Report.from_json(path))

    comparison = get_acc_report_df(reports).set_index('Model')
    assert comparison['Score'].to_dict() == {'Llama-3.1-70B-Instruct': 0.5270, 'Qwen2.5-14B-Instruct': 0.5222}
    assert comparison['Num'].tolist() == [3400, 3400]


def test_partial_run_averages_only_subsets_with_f1(tmp_path: Path) -> None:
    adapter = get_benchmark('process_bench', TaskConfig(datasets=['process_bench']))
    scores = {
        'gsm8k': [{'correct_acc': 1}, {'error_acc': 1}],
        'math': [{'correct_acc': 0}] * 3 + [{'error_acc': 0}],
        'omnimath': [{'error_acc': 1}],
    }
    subset_scores = {
        subset: adapter.aggregate_scores([SampleScore(score=Score(value=row)) for row in rows])
        for subset, rows in scores.items()
    }
    report = adapter.generate_report(subset_scores, 'model', str(tmp_path))
    assert report.score == 0.5
    assert report.num == 6
    metrics = {metric.identity.name: metric.score for metric in report.metrics}
    assert metrics['correct_acc'] == 0.25


def test_single_label_run_marks_process_bench_primary_as_unavailable() -> None:
    """A sampled run with one reference label must not invent an F1 score."""
    adapter = ProcessBenchAdapter(
        benchmark_meta=BenchmarkMeta(
            name='process_bench',
            pretty_name='ProcessBench',
            description='',
            dataset_id='Qwen/ProcessBench',
            metric_list=['error_acc', 'correct_acc', 'simple_f1_score'],
            primary_metric='simple_f1_score',
            aggregation='f1',
        ),
        task_config=TaskConfig(datasets=['process_bench']),
    )
    aggregate_scores = adapter.aggregate_scores([
        SampleScore(sample_id=1, score=Score(value={'error_acc': 1.0})),
    ])

    report = adapter.generate_report({'gsm8k': aggregate_scores}, model_name='model', output_dir='')

    assert [score.metric_name for score in aggregate_scores] == ['error_acc']
    assert report.primary_metric is None
    assert report.score is None
    assert report.primary_metric_unavailable_reason is not None
