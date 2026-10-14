"""Needle-in-a-Haystack reports one stable accuracy metric per (context length, depth) cell.

The cell is carried by the ``context_length`` and ``depth_percent`` identity dimensions, the overall
mean is the primary metric, and the heatmap reads its cells from those dimensions instead of parsing
a display name. Reports written before the metric semantics contract still load with the same cells.
"""
import json
import os
from typing import Dict, List, Tuple

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import MetricIdentity, MetricKind, SampleScore, Score
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.needle_haystack import needle_haystack_adapter
from evalscope.config import TaskConfig
from evalscope.constants import JudgeScoreType, ScoreStatus
from evalscope.metrics.judge.llm_judge import DEFAULT_PROMPT_TEMPLATE, LLMJudge
from evalscope.report import Report

NEEDLE = '\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n'
QUESTION = 'What is the best thing to do in San Francisco?'
OVERALL = MetricIdentity(name='accuracy', aggregation='mean', dimensions={'scope': 'overall'})

# (context_length, depth_percent) -> predictions per subset; one sample per prediction.
SINGLE_CELL = {(1000, 0): {'english': [NEEDLE], 'chinese': ['no idea']}}
GRID = {
    (1000, 0): {'english': [NEEDLE, NEEDLE], 'chinese': [NEEDLE, 'no idea']},
    (1000, 50): {'english': [NEEDLE, 'no idea'], 'chinese': [NEEDLE, NEEDLE]},
    (2000, 0): {'english': ['no idea', 'no idea'], 'chinese': [NEEDLE, 'no idea']},
    (2000, 50): {'english': [NEEDLE, 'no idea'], 'chinese': ['no idea', 'no idea']},
}


class ScriptedJudge:
    score_type = JudgeScoreType.PATTERN
    score_mapping = {'A': 1.0, 'B': 0.0}
    prompt_template = DEFAULT_PROMPT_TEMPLATE
    system_prompt = None
    build_prompt = LLMJudge.build_prompt

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.judge_id = self.model_id = 'scripted'
        self.calls: List[object] = []

    def generate(self, messages):
        self.calls.append(messages)
        return ModelOutput.from_content('scripted', self.reply)


def _adapter(monkeypatch: pytest.MonkeyPatch, strategy: str = 'rule'):
    # Scoring and report building do not need the optional plotting dependencies.
    monkeypatch.setattr(needle_haystack_adapter, 'check_import', lambda *args, **kwargs: True)
    config = TaskConfig(model='mock', datasets=['needle_haystack'], judge={'strategy': strategy})
    return get_benchmark('needle_haystack', config)


def _state(sample_id: int, prediction: str, context_length: int, depth_percent: int) -> TaskState:
    sample = Sample(
        id=sample_id,
        group_id=sample_id,
        input=QUESTION,
        target=NEEDLE,
        metadata={'context': '', 'context_length': context_length, 'depth_percent': depth_percent},
    )
    return TaskState(
        model='mock', sample=sample, output=ModelOutput.from_content('mock', prediction), completed=True
    )


def _cell(context_length: int, depth_percent: int) -> MetricIdentity:
    return MetricIdentity(
        name='accuracy',
        aggregation='mean',
        dimensions={'context_length': context_length, 'depth_percent': depth_percent},
    )


def _score_subsets(adapter, grid: Dict[Tuple[int, int], Dict[str, List[str]]]) -> Dict[str, List[SampleScore]]:
    """Score every prediction in ``grid`` through the adapter, grouped by subset."""
    sample_scores: Dict[str, List[SampleScore]] = {}
    sample_id = 0
    for (context_length, depth_percent), predictions_by_subset in grid.items():
        for subset, predictions in predictions_by_subset.items():
            for prediction in predictions:
                state = _state(sample_id, prediction, context_length, depth_percent)
                sample_scores.setdefault(subset, []).append(adapter.calculate_metrics(state))
                sample_id += 1
    return sample_scores


def _expected_accuracy(predictions: List[str]) -> float:
    return sum(prediction == NEEDLE for prediction in predictions) / len(predictions)


def _subset_scores(report: Report, identity: MetricIdentity) -> Dict[str, Tuple[float, int]]:
    metric = next(metric for metric in report.metrics if metric.identity == identity)
    return {subset.name: (subset.score, subset.num) for category in metric.categories for subset in category.subsets}


def test_evaluation_version_marks_the_new_metric_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1.1 reports one accuracy metric with the cells as dimensions, instead of one metric per cell."""
    assert _adapter(monkeypatch)._benchmark_meta.evaluation_version == 'v1.1'


def test_rule_score_uses_a_stable_metric_name(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch, strategy='rule')

    score = adapter.calculate_metrics(_state(0, NEEDLE, 1000, 0)).score

    assert score.value == {'acc': 1.0}
    assert score.main_score_name == 'acc'


def test_judge_score_uses_the_same_metric_name_as_the_rule_score(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch, strategy='llm')
    judge = ScriptedJudge(json.dumps({'reasoning': 'Mostly aligned.', 'verdict': 7}))
    adapter.llm_judge = judge

    score = adapter.calculate_metrics(_state(0, 'Eat a sandwich in Dolores Park.', 32000, 100)).score

    assert len(judge.calls) == 1
    assert score.status is ScoreStatus.SUCCESS
    assert score.value == {'acc': pytest.approx(0.7)}
    assert score.main_score_name == 'acc'


def test_unusable_judge_reply_falls_back_to_the_rule_score(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch, strategy='llm')
    adapter.llm_judge = ScriptedJudge('not JSON')

    score = adapter.calculate_metrics(_state(0, NEEDLE, 1000, 0)).score

    assert score.status is ScoreStatus.DEGRADED
    assert score.value == {'acc': 1.0}


@pytest.mark.parametrize('grid', [SINGLE_CELL, GRID], ids=['single_cell', 'lengths_x_depths'])
def test_report_keeps_cells_as_dimensions_and_selects_the_overall_mean(
    monkeypatch: pytest.MonkeyPatch, tmp_path, grid: Dict[Tuple[int, int], Dict[str, List[str]]]
) -> None:
    adapter = _adapter(monkeypatch)
    sample_scores = _score_subsets(adapter, grid)
    score_dict = {subset: adapter.aggregate_scores(scores) for subset, scores in sample_scores.items()}

    report = adapter.generate_report(score_dict, model_name='mock', output_dir=str(tmp_path))

    assert {metric.identity for metric in report.metrics} == {OVERALL} | {_cell(*cell) for cell in grid}
    assert all(metric.semantics.kind is MetricKind.QUALITY for metric in report.metrics)
    assert report.primary_metric_identity == OVERALL
    assert report.primary_metric_unavailable_reason is None
    assert report.primary_metric.semantics.semantic_id == 'quality.accuracy.ratio'

    every_prediction = [prediction for cell in grid.values() for subset in cell.values() for prediction in subset]
    assert report.score == pytest.approx(_expected_accuracy(every_prediction), abs=1e-4)
    for cell, predictions_by_subset in grid.items():
        assert _subset_scores(report, _cell(*cell)) == {
            subset: (pytest.approx(_expected_accuracy(predictions), abs=1e-4), len(predictions))
            for subset, predictions in predictions_by_subset.items()
        }


def test_excluded_samples_are_left_out_of_every_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(monkeypatch)
    scored = _score_subsets(adapter, {(1000, 0): {'english': [NEEDLE, 'no idea']}})['english']

    def excluded(sample_id: int, context_length: int, depth_percent: int) -> SampleScore:
        return SampleScore(
            score=Score(status=ScoreStatus.EXCLUDED),
            sample_id=sample_id,
            group_id=sample_id,
            sample_metadata={'context_length': context_length, 'depth_percent': depth_percent},
        )

    # One excluded sample joins a scored cell, and another is the only sample of its cell.
    aggregates = adapter.aggregate_scores(scored + [excluded(10, 1000, 0), excluded(11, 2000, 50)])

    by_identity = {aggregate.identity: (aggregate.score, aggregate.num) for aggregate in aggregates}
    assert by_identity == {OVERALL: (0.5, 2), _cell(1000, 0): (0.5, 2)}
    assert adapter.aggregate_scores([excluded(12, 1000, 0)]) == []
    assert adapter.aggregate_scores([]) == []


def test_heatmap_is_drawn_from_dimensions_for_every_subset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip('matplotlib')
    pytest.importorskip('seaborn')
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    adapter = _adapter(monkeypatch)
    sample_scores = _score_subsets(adapter, GRID)
    score_dict = {subset: adapter.aggregate_scores(scores) for subset, scores in sample_scores.items()}

    report = adapter.generate_report(score_dict, model_name='mock', output_dir=str(tmp_path))

    for subset in ('english', 'chinese'):
        assert os.path.isfile(tmp_path / f'needle_haystack_heatmap_{subset}.png')

    from evalscope.benchmarks.needle_haystack.utils import collect_cell_scores

    cells = collect_cell_scores(report)
    english = cells[cells['Subset'] == 'english'].pivot(index='Depth', columns='Context', values='Score')
    assert english.index.tolist() == [0, 50]
    assert english.columns.tolist() == [1000, 2000]
    assert english.loc[50, 1000] == pytest.approx(0.5)
    assert english.loc[0, 2000] == pytest.approx(0.0)


def _legacy_metric(name: str, subset_scores: Dict[str, float]) -> dict:
    subsets = [{'name': subset, 'score': score, 'num': 1} for subset, score in subset_scores.items()]
    return {'name': name, 'num': len(subsets), 'score': 0.0, 'categories': [{'name': ['default'], 'subsets': subsets}]}


def _legacy_report() -> Report:
    """A report written before the semantics contract, which spelled each cell ``Context#<n> Depth#<d>``."""
    return Report.from_dict({
        'name': 'mock@needle_haystack',
        'dataset_name': 'needle_haystack',
        'model_name': 'mock',
        'score': 1.0,
        'metrics': [
            _legacy_metric('Context#1000 Depth#0', {'english': 1.0, 'chinese': 0.0}),
            _legacy_metric('Context#32000 Depth#100', {'english': 0.1, 'chinese': 1.0}),
        ],
    })


def test_legacy_report_cells_are_read_as_dimensions() -> None:
    from evalscope.benchmarks.needle_haystack.utils import collect_cell_scores

    report = _legacy_report()

    assert [metric.identity for metric in report.metrics] == [_cell(1000, 0), _cell(32000, 100)]
    assert all(metric.semantics.kind is MetricKind.QUALITY for metric in report.metrics)
    cells = collect_cell_scores(report)
    assert sorted(cells.itertuples(index=False, name=None)) == [
        ('chinese', 1000, 0, 0.0),
        ('chinese', 32000, 100, 1.0),
        ('english', 1000, 0, 1.0),
        ('english', 32000, 100, 0.1),
    ]


def test_heatmap_is_drawn_from_a_legacy_report(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip('matplotlib')
    pytest.importorskip('seaborn')
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    adapter = _adapter(monkeypatch)

    adapter._on_generate_report_end(_legacy_report(), str(tmp_path))

    for subset in ('english', 'chinese'):
        assert os.path.isfile(tmp_path / f'needle_haystack_heatmap_{subset}.png')
