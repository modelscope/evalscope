from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.metric import SampleScore, Score
from evalscope.benchmarks.process_bench.process_bench_adapter import ProcessBenchAdapter
from evalscope.config import TaskConfig


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
