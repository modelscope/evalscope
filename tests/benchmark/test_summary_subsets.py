"""Summary rows (OVERALL, EN/CN, NON_LIVE, ...) that adapters append in ``_on_generate_report_end``
are derived from the real subsets, so they are flagged ``is_aggregate`` and stay out of the
report's sample count and score, including after the report is written and read back."""
import importlib
import os

import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.report import get_report_list

# (benchmark, module whose optional-dependency check is skipped: report building does not need it)
CASES = [
    ('bfcl_v3', 'evalscope.benchmarks.bfcl.v3.bfcl_v3_adapter'),
    ('acebench', None),
    ('ocr_bench_v2', 'evalscope.benchmarks.ocr_bench.ocr_bench_v2.ocr_bench_v2_adapter'),
]
N = 10


@pytest.mark.parametrize(('name', 'module'), CASES)
def test_summary_rows_do_not_count_toward_totals(name, module, monkeypatch, tmp_path):
    if module:
        monkeypatch.setattr(importlib.import_module(module), 'check_import', lambda *a, **k: True)
    adapter = get_benchmark(name, TaskConfig(model='mock', datasets=[name]))
    # Subset i gets i % 9 + 1 of its N samples right.
    values = {s: [1.0] * (i % 9 + 1) + [0.0] * (N - i % 9 - 1) for i, s in enumerate(adapter.subset_list)}
    score_dict = {
        s: adapter.aggregate_scores(
            [SampleScore(score=Score(value={'acc': v}), sample_id=j, group_id=j) for j, v in enumerate(vals)]
        )
        for s, vals in values.items()
    }
    report = adapter.generate_report(score_dict, model_name='mock', output_dir=str(tmp_path))

    report.to_json(os.path.join(tmp_path, 'reports', 'mock', 'report.json'))
    (reloaded, ) = get_report_list([str(tmp_path)])

    num_samples = N * len(values)
    expected = sum(map(sum, values.values())) / num_samples
    assert report.to_dict()['num'] == num_samples
    assert reloaded.primary_metric.num == num_samples
    assert reloaded.score == pytest.approx(expected, abs=1e-4)
