"""BFCL-v3 summary rows must use the official leaderboard formula (``generate_leaderboard_csv``)."""
import importlib

import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

# (correct, total) per category; live_multiple is larger so the live row must be count-weighted.
RESULTS = {
    'simple': (10, 10),
    'java': (7, 10),
    'javascript': (7, 10),
    'multiple': (9, 10),
    'parallel': (9, 10),
    'parallel_multiple': (9, 10),
    'irrelevance': (5, 10),
    'live_simple': (8, 10),
    'live_multiple': (6, 20),
    'live_parallel': (9, 10),
    'live_parallel_multiple': (7, 10),
    'live_irrelevance': (9, 10),
    'live_relevance': (8, 10),
    'multi_turn_base': (5, 10),
    'multi_turn_miss_func': (2, 10),
    'multi_turn_miss_param': (2, 10),
    'multi_turn_long_context': (4, 10),
}


def _acc(name: str) -> float:
    correct, total = RESULTS[name]
    return correct / total


def _official_rows() -> dict:
    simple_ast = (_acc('simple') + _acc('java') + _acc('javascript')) / 3
    non_live = (simple_ast + _acc('multiple') + _acc('parallel') + _acc('parallel_multiple') + _acc('irrelevance')) / 5
    live_names = [name for name in RESULTS if name.startswith('live_')]
    live = sum(RESULTS[name][0] for name in live_names) / sum(RESULTS[name][1] for name in live_names)
    multi_turn = sum(_acc(name) for name in RESULTS if name.startswith('multi_turn_')) / 4
    return {'NON_LIVE': non_live, 'LIVE': live, 'MULTI_TURN': multi_turn, 'OVERALL': (non_live + live + multi_turn) / 3}


def test_summary_rows_match_official_leaderboard(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    module = importlib.import_module('evalscope.benchmarks.bfcl.v3.bfcl_v3_adapter')
    monkeypatch.setattr(module, 'check_import', lambda *args, **kwargs: True)
    adapter = get_benchmark('bfcl_v3', TaskConfig(model='mock', datasets=['bfcl_v3']))
    score_dict = {}
    for name, (correct, total) in RESULTS.items():
        values = [1.0] * correct + [0.0] * (total - correct)
        scores = [SampleScore(score=Score(value={'acc': v}), sample_id=i, group_id=i) for i, v in enumerate(values)]
        score_dict[name] = adapter.aggregate_scores(scores)

    report = adapter.generate_report(score_dict, model_name='mock', output_dir=str(tmp_path))

    rows = {
        subset.name: subset.score
        for category in report.metrics[0].categories for subset in category.subsets if subset.is_aggregate
    }
    for name, expected in _official_rows().items():
        assert rows[name] == pytest.approx(expected, abs=1e-4), name
