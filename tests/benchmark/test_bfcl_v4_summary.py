import pytest

pytest.importorskip('bfcl_eval')

from bfcl_eval.eval_checker.eval_runner_helper import get_category_score

from evalscope.benchmarks.bfcl.v4.utils import ALL_SCORING_CATEGORIES, compute_aggregate_subsets
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.metrics.semantics.naming import canonicalize_producer_identity
from evalscope.report import Subset
from evalscope.report.report import Category, Metric, Report


def _report(subsets: dict[str, tuple[float, int]]) -> Report:
    identity = canonicalize_producer_identity('acc', 'mean')
    semantics = get_semantics_resolver().resolve('bfcl_v4', identity).semantics
    category = Category(
        name='default',
        subsets=[Subset(name=name, score=score, num=num) for name, (score, num) in subsets.items()],
    )
    metric = Metric(identity=identity, semantics=semantics, categories=[category])
    return Report(name='mock@bfcl_v4', metrics=[metric], dataset_name='bfcl_v4', model_name='mock')


def _summary_rows(report: Report) -> dict[str, Subset]:
    compute_aggregate_subsets(report)
    return {
        subset.name: subset
        for category in report.metrics[0].categories
        for subset in category.subsets
        if subset.is_aggregate
    }


def _expected_live_simple_only_score() -> float:
    score_dict = {'live_simple': {'accuracy': 0.8, 'total_count': 10, 'display_accuracy': 0.8}}
    names = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple']
    total_count = sum(get_category_score(score_dict, name)['total_count'] for name in names)
    return 0.8 * 10 / total_count


def test_full_evaluation_is_unchanged() -> None:
    report = _report({name: (0.65, 10) for name in ALL_SCORING_CATEGORIES})

    rows = _summary_rows(report)

    for name in ['AGENTIC', 'MULTI_TURN', 'NON_LIVE', 'LIVE', 'HALLUCINATION', 'OVERALL']:
        assert rows[name].score == pytest.approx(0.65)
    assert rows['OVERALL'].num == 10 * (len(ALL_SCORING_CATEGORIES) - 1)


def test_only_complete_non_live_group_keeps_its_official_weight() -> None:
    names = ['simple_python', 'simple_java', 'simple_javascript', 'multiple', 'parallel', 'parallel_multiple']
    report = _report({name: (0.8, 100) for name in names})

    rows = _summary_rows(report)

    assert rows['NON_LIVE'].score == pytest.approx(0.8)
    assert rows['OVERALL'].score == pytest.approx(0.08)
    assert rows['OVERALL'].num == 600


def test_missing_subcategories_are_zero_in_unweighted_groups() -> None:
    report = _report({'simple_python': (0.8, 100)})

    rows = _summary_rows(report)

    assert rows['NON_LIVE'].score == pytest.approx(round(0.8 / 3 / 4, 4))
    assert rows['OVERALL'].score == pytest.approx(round(0.8 / 3 / 4 / 10, 4))
    assert rows['OVERALL'].num == 100


def test_live_weighting_uses_full_counts_for_missing_categories() -> None:
    report = _report({'live_simple': (0.8, 10)})

    rows = _summary_rows(report)

    expected_live = _expected_live_simple_only_score()
    assert rows['LIVE'].score == pytest.approx(round(expected_live, 4))
    assert rows['OVERALL'].score == pytest.approx(round(expected_live / 10, 4))
    assert rows['OVERALL'].num == 10


def test_zero_num_subset_is_treated_as_unevaluated() -> None:
    report = _report({'live_simple': (0.8, 10), 'live_multiple': (0.0, 0)})

    rows = _summary_rows(report)

    expected_live = _expected_live_simple_only_score()
    assert rows['LIVE'].score == pytest.approx(round(expected_live, 4))
    assert rows['OVERALL'].score == pytest.approx(round(expected_live / 10, 4))
    assert rows['OVERALL'].num == 10
