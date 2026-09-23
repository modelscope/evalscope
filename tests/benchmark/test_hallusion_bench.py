"""Regression tests for HallusionBench figure/question grouping.

These mirror the official HallusionBench evaluation
(https://github.com/tianyi-lab/HallusionBench, ``utils.py``):

- The figure-level (``fAcc``) and question-pair (``qAcc``) grouping keys join
  ``category`` with ``subcategory`` / ``set_id`` / ``figure_id`` / ``question_id``.
  The two categories (``VD`` / ``VS``) reuse the same
  subcategory/set_id/figure_id/question_id numbering (e.g. ``ocr_0_0`` and
  ``chart_0_1`` both appear under VD *and* VS), so dropping ``category`` merges
  distinct figures/questions across categories.
- Figure-level accuracy skips VS "no-figure" records (``figure_id == 0``).
"""

from typing import List

from evalscope.api.metric.scorer import SampleScore, Score
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.hallusion_bench.hallusion_bench_adapter import HallusionBenchAdapter


def _adapter() -> HallusionBenchAdapter:
    return HallusionBenchAdapter(benchmark_meta=BENCHMARK_REGISTRY['hallusion_bench'])


def _sample_score(category, subcategory, set_id, figure_id, question_id, acc) -> SampleScore:
    return SampleScore(
        score=Score(value={'acc': acc}, main_score_name='acc'),
        sample_metadata={
            'category': category,
            'subcategory': subcategory,
            'set_id': set_id,
            'figure_id': figure_id,
            'question_id': question_id,
        },
    )


def _overall(scores: List[SampleScore]):
    aggregates = _adapter().aggregate_scores(scores)
    return {
        agg.dimensions['target']: (round(agg.score, 4), agg.num)
        for agg in aggregates
        if agg.dimensions.get('level') == 'overall'
    }


def test_grouping_key_separates_categories():
    """VD and VS share figure_id/question_id numbering; they must not be merged.

    Both records here use ``figure_id == '1'`` so the VS "no-figure" skip does not
    apply, isolating the effect of the ``category`` component of the key. Without
    ``category`` in the key both rows collapse into a single ``chart_0_1`` group
    that is neither all-correct (fAcc/qAcc = 0.0); with it they form two groups,
    one correct (fAcc/qAcc = 0.5).
    """
    scores = [
        _sample_score('VD', 'chart', '0', '1', '1', 1),
        _sample_score('VS', 'chart', '0', '1', '1', 0),
    ]

    overall = _overall(scores)

    assert overall['answer'] == (0.5, 2)
    assert overall['figure'] == (0.5, 2)
    assert overall['question'] == (0.5, 2)


def test_figure_accuracy_skips_vs_no_figure_records():
    """VS records with ``figure_id == '0'`` carry no figure and are excluded from fAcc.

    They still count toward answer-level (aAcc) and question-level (qAcc) accuracy.
    """
    scores = [
        _sample_score('VD', 'ocr', '0', '1', '0', 1),  # real figure, correct
        _sample_score('VS', 'ocr', '5', '0', '0', 0),  # no figure (figure_id == 0), wrong
    ]

    overall = _overall(scores)

    # answer- and question-level accuracy see both records.
    assert overall['answer'] == (0.5, 2)
    assert overall['question'] == (0.5, 2)
    # figure-level accuracy drops the VS no-figure record, leaving one correct figure.
    assert overall['figure'] == (1.0, 1)
