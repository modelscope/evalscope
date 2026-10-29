"""OCRBench-v2 EN and CN weight their categories equally and OVERALL is (EN + CN) / 2, as in
``utils.ocrbench_v2_aggregate_accuracy``, whatever the number of samples per category."""
import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.ocr_bench.ocr_bench_v2 import ocr_bench_v2_adapter
from evalscope.config import TaskConfig


def test_overall_is_the_mean_of_en_and_cn(monkeypatch, tmp_path):
    # Report building does not need the optional OCRBench-v2 scoring dependencies.
    monkeypatch.setattr(ocr_bench_v2_adapter, 'check_import', lambda *a, **k: True)
    adapter = get_benchmark('ocr_bench_v2', TaskConfig(model='mock', datasets=['ocr_bench_v2']))
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
