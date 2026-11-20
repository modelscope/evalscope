"""OCRBench-v2 EN and CN weight their categories equally and OVERALL is (EN + CN) / 2, as in
``utils.ocrbench_v2_aggregate_accuracy``, whatever the number of samples per category."""
from pathlib import Path

import pytest

from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.ocr_bench.ocr_bench_v2 import ocr_bench_v2_adapter
from evalscope.config import TaskConfig
from evalscope.report import Report
from evalscope.report.data_frames import get_acc_report_df


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> ocr_bench_v2_adapter.OCRBenchV2Adapter:
    # Report building does not need the optional OCRBench-v2 scoring dependencies.
    monkeypatch.setattr(ocr_bench_v2_adapter, 'check_import', lambda *a, **k: True)
    return get_benchmark('ocr_bench_v2', TaskConfig(model='mock', datasets=['ocr_bench_v2']))


def test_overall_is_the_mean_of_en_and_cn(adapter: ocr_bench_v2_adapter.OCRBenchV2Adapter, tmp_path: Path) -> None:
    # 10 samples per subset. EN subsets score 0.6, except 'text spotting en' (a category of its
    # own) at 1.0; CN subsets score 0.4.
    score_dict = {}
    for subset in adapter.subset_list:
        correct = 10 if subset == 'text spotting en' else 6 if subset.endswith(' en') else 4
        score_dict[subset] = adapter.aggregate_scores([
            SampleScore(score=Score(value={'acc': float(i < correct)}), sample_id=i, group_id=i) for i in range(10)
        ])
    report = adapter.generate_report(score_dict, model_name='mock', output_dir=str(tmp_path))

    summary = {s.name: s.score for c in report.metrics[0].categories for s in c.subsets}
    # 8 EN categories: seven at 0.6 and text_spotting_en at 1.0.
    assert summary['EN'] == pytest.approx((7 * 0.6 + 1.0) / 8)
    assert summary['CN'] == pytest.approx(0.4)
    assert summary['OVERALL'] == pytest.approx(((7 * 0.6 + 1.0) / 8 + 0.4) / 2)
    assert report.score == summary['OVERALL']


def test_compare_ranks_models_by_overall_after_reload(
    adapter: ocr_bench_v2_adapter.OCRBenchV2Adapter, tmp_path: Path
) -> None:
    reports = []
    for model, en, cn in [('english_only', 1.0, 0.0), ('balanced', 0.6, 0.6)]:
        scores = {
            'text spotting en': [AggScore(metric_name='acc', aggregation='mean', score=en, num=15)],
            'full-page OCR cn': [AggScore(metric_name='acc', aggregation='mean', score=cn, num=5)],
        }
        report = adapter.generate_report(scores, model_name=model, output_dir=str(tmp_path))
        path = str(tmp_path / f'{model}.json')
        report.to_json(path)
        reports.append(Report.from_json(path))

    comparison = get_acc_report_df(reports).set_index('Model')
    assert comparison['Score'].to_dict() == {'english_only': 0.5, 'balanced': 0.6}
    assert comparison['Num'].to_dict() == {'english_only': 20, 'balanced': 20}


def test_single_language_uses_available_categories(
    adapter: ocr_bench_v2_adapter.OCRBenchV2Adapter, tmp_path: Path
) -> None:
    scores = {
        'text spotting en': [AggScore(metric_name='acc', aggregation='mean', score=1.0, num=9)],
        'text grounding en': [AggScore(metric_name='acc', aggregation='mean', score=0.0, num=1)],
    }
    report = adapter.generate_report(scores, model_name='mock', output_dir=str(tmp_path))
    assert report.score == 0.5
    assert report.num == 10
