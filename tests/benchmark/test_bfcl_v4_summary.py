"""BFCL-v4 OVERALL must only weigh the summary groups that were actually evaluated."""

import pytest

from evalscope.benchmarks.bfcl.v4.utils import compute_aggregate_subsets
from evalscope.metrics.semantics import get_semantics_resolver
from evalscope.metrics.semantics.naming import canonicalize_producer_identity
from evalscope.report import Subset, percentage_weighted_average_from_subsets
from evalscope.report.report import Category, Metric, Report


def _report(subsets: dict) -> Report:
    """Build a report that holds one real subset per ``name -> (score, num)`` entry."""
    identity = canonicalize_producer_identity('acc', 'mean')
    semantics = get_semantics_resolver().resolve('bfcl_v4', identity).semantics
    category = Category(
        name='default', subsets=[Subset(name=name, score=score, num=num) for name, (score, num) in subsets.items()]
    )
    metric = Metric(identity=identity, semantics=semantics, categories=[category])
    return Report(name='mock@bfcl_v4', metrics=[metric], dataset_name='bfcl_v4', model_name='mock')


def _summary_rows(report: Report) -> dict:
    return {
        subset.name: subset.score
        for category in report.metrics[0].categories for subset in category.subsets if subset.is_aggregate
    }


def test_overall_ignores_groups_without_samples() -> None:
    # Only the non-live groups were run (a ``subset_list`` that leaves the other groups out).
    report = _report({'simple_python': (0.8, 100), 'multiple': (0.8, 100), 'parallel': (0.8, 100)})

    compute_aggregate_subsets(report)

    rows = _summary_rows(report)
    assert rows['NON_LIVE'] == pytest.approx(0.8)
    assert rows['OVERALL'] == pytest.approx(0.8)


def test_overall_renormalizes_weights_over_evaluated_groups() -> None:
    report = _report({
        'simple_python': (1.0, 10),
        'live_simple': (0.5, 10),
        'multi_turn_base': (0.0, 10),
    })

    compute_aggregate_subsets(report)

    rows = _summary_rows(report)
    # non_live 1.0 (weight 10), live 0.5 (weight 10), multi_turn 0.0 (weight 30); the other groups are absent.
    assert rows['OVERALL'] == pytest.approx((1.0 * 10 + 0.5 * 10 + 0.0 * 30) / 50)


def test_overall_unchanged_when_every_group_is_evaluated() -> None:
    report = _report({
        'simple_python': (0.9, 10),
        'live_simple': (0.7, 10),
        'irrelevance': (0.6, 10),
        'multi_turn_base': (0.2, 10),
        'web_search_base': (0.4, 10),
        'memory_kv': (0.6, 10),
    })

    compute_aggregate_subsets(report)

    rows = _summary_rows(report)
    # agentic = mean(web_search 0.4, memory 0.6) = 0.5; non_live 0.9; live 0.7; multi_turn 0.2; hallucination 0.6.
    expected = 0.4 * 0.5 + 0.3 * 0.2 + 0.1 * 0.9 + 0.1 * 0.7 + 0.1 * 0.6
    assert rows['OVERALL'] == pytest.approx(expected)


def test_percentage_weighted_average_skips_empty_subsets() -> None:
    subset_dict = {'a': Subset(name='a', score=0.6, num=5), 'b': Subset(name='b', score=0.0, num=0)}

    result = percentage_weighted_average_from_subsets(['a', 'b'], subset_dict, weights=[30, 70])

    assert result.score == pytest.approx(0.6)
    assert result.num == 5
