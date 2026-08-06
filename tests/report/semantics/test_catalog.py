"""Tests for the central metric semantics catalog.

Covers the three catalog tables and their import time validation:

* ``TestCatalogEntriesResolve`` -- Property 6: every entry materializes into a legal semantics.
* ``TestBaselineReferences`` -- Property 14: no dangling ``baseline`` reference.
* ``TestNoAggregationGroups`` -- Property 19: v1 declares no cross-benchmark aggregation.
* ``TestImportTimeValidation`` -- an illegal entry or a dangling baseline aborts validation.
* ``TestLookupMetricEntry`` -- exact-key lookup without normalization.
"""

import pytest
from typing import Dict

from evalscope.api.metric.semantics import MetricDirection, MetricDisplayKind, MetricEntry, MetricRole, MetricSemantics
from evalscope.metrics.semantics import catalog as catalog_module
from evalscope.metrics.semantics.baselines import SEMANTIC_BASELINES
from evalscope.metrics.semantics.catalog import (
    BENCHMARK_DYNAMIC_METRICS,
    BENCHMARK_METRIC_OVERRIDES,
    METRIC_NAME_SEMANTICS,
    lookup_metric_entry,
)


def _all_entries() -> Dict[str, MetricEntry]:
    """Return every catalog entry keyed by the final report metric name it is declared for."""
    entries: Dict[str, MetricEntry] = dict(METRIC_NAME_SEMANTICS)
    for (_, metric_name), entry in BENCHMARK_METRIC_OVERRIDES.items():
        entries[metric_name] = entry
    return entries


class TestCatalogEntriesResolve:
    """Feature: metric-semantics-governance, Property 6: every catalog entry materializes into a
    MetricSemantics that passes all contract validation rules."""

    @pytest.mark.parametrize('metric_name', sorted(METRIC_NAME_SEMANTICS))
    def test_metric_name_entry_resolves(self, metric_name: str) -> None:
        semantics = METRIC_NAME_SEMANTICS[metric_name].resolve(metric_name)

        assert isinstance(semantics, MetricSemantics)
        assert semantics.semantic_id
        assert semantics.metric_name

    @pytest.mark.parametrize('key', sorted(BENCHMARK_METRIC_OVERRIDES))
    def test_override_entry_resolves(self, key) -> None:
        _, metric_name = key
        assert isinstance(BENCHMARK_METRIC_OVERRIDES[key].resolve(metric_name), MetricSemantics)

    def test_diagnostic_entries_carry_no_direction_nor_comparison_group(self) -> None:
        for metric_name, entry in _all_entries().items():
            semantics = entry.resolve(metric_name)
            if semantics.role is MetricRole.DIAGNOSTIC:
                assert semantics.direction is MetricDirection.NONE, metric_name
                assert semantics.comparison_group is None, metric_name

    def test_percent_entries_declare_range_and_multiplier(self) -> None:
        for metric_name, entry in _all_entries().items():
            semantics = entry.resolve(metric_name)
            if semantics.display_kind is MetricDisplayKind.PERCENT:
                assert semantics.value_range is not None, metric_name
                assert semantics.display_multiplier is not None, metric_name


class TestBaselineReferences:
    """Feature: metric-semantics-governance, Property 14: every declared baseline reference
    exists in the baseline table, otherwise importing the catalog fails."""

    def test_no_dangling_baseline_in_metric_name_table(self) -> None:
        dangling = {
            name: entry.baseline
            for name, entry in METRIC_NAME_SEMANTICS.items()
            if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES
        }

        assert dangling == {}

    def test_no_dangling_baseline_in_override_table(self) -> None:
        dangling = {
            key: entry.baseline
            for key, entry in BENCHMARK_METRIC_OVERRIDES.items()
            if entry.baseline is not None and entry.baseline not in SEMANTIC_BASELINES
        }

        assert dangling == {}

    def test_expected_baselines_are_available(self) -> None:
        for baseline_id in ('quality.accuracy.ratio', 'quality.wer.ratio', 'perf.latency.seconds'):
            assert baseline_id in SEMANTIC_BASELINES


class TestNoAggregationGroups:
    """Feature: metric-semantics-governance, Property 19: v1 declares no aggregation group, so
    no cross-benchmark total is ever produced."""

    def test_catalog_declares_no_aggregation_group(self) -> None:
        declared = {
            metric_name
            for metric_name, entry in _all_entries().items() if entry.resolve(metric_name).aggregation_group is not None
        }

        assert declared == set()


class TestImportTimeValidation:
    """An illegal entry or a dangling baseline must abort the catalog validation."""

    def test_dangling_baseline_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(METRIC_NAME_SEMANTICS, 'bogus_metric', MetricEntry(baseline='quality.does.not.exist'))

        with pytest.raises(ValueError, match='unknown baseline'):
            catalog_module._validate_catalog()

    def test_illegal_entry_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # role=primary with direction=none violates R1 and must not resolve.
        monkeypatch.setitem(
            METRIC_NAME_SEMANTICS,
            'bogus_metric',
            MetricEntry(
                semantic_id='quality.bogus.ratio',
                role=MetricRole.PRIMARY,
                direction=MetricDirection.NONE,
            ),
        )

        with pytest.raises(ValueError):
            catalog_module._validate_catalog()

    def test_shipped_catalog_validates(self) -> None:
        catalog_module._validate_catalog()


class TestLookupMetricEntry:
    """Lookups are exact-key: no normalization, no fuzzy matching."""

    def test_declared_name_is_found(self) -> None:
        assert lookup_metric_entry('mean_acc') is not None

    @pytest.mark.parametrize('variant', ['MEAN_ACC', 'meanacc', 'mean_acc ', ' mean_acc', 'mean_acc_v2'])
    def test_name_variants_are_not_found(self, variant: str) -> None:
        assert lookup_metric_entry(variant) is None

    def test_dynamic_table_is_a_mapping_of_allow_lists(self) -> None:
        assert isinstance(BENCHMARK_DYNAMIC_METRICS, dict)
        for benchmark_name, allowed in BENCHMARK_DYNAMIC_METRICS.items():
            assert isinstance(benchmark_name, str)
            assert all(isinstance(name, str) for name in allowed)


class TestGsm8kAccuracy:
    """GSM8K reports a single ``mean_acc`` metric that must resolve to a primary Accuracy."""

    def test_mean_acc_resolves_to_primary_accuracy(self) -> None:
        semantics = METRIC_NAME_SEMANTICS['mean_acc'].resolve('mean_acc')

        assert semantics.semantic_id == 'quality.accuracy.ratio'
        assert semantics.role is MetricRole.PRIMARY
        assert semantics.direction is MetricDirection.HIGHER_IS_BETTER
