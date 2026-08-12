"""Tests for the central metric semantics catalog.

``catalog.py`` calls ``_validate_catalog()`` at import time, which resolves **every** entry through
the full ``MetricSemantics`` contract. A dangling baseline or an invalid entry therefore
makes the module unimportable and this file error at collection. Tests that merely re-assert
"every entry resolves" / "no baseline dangles" cannot fail independently and were removed; what is
left exercises the failure path explicitly (via monkeypatch) and pins concrete semantic choices no
validator enforces.
"""

import pytest

from evalscope.api.metric.semantics import MetricDirection, MetricRole
from evalscope.metrics.semantics import catalog as catalog_module
from evalscope.metrics.semantics.catalog import METRIC_DEFINITIONS
from evalscope.metrics.semantics.entry import MetricEntry


class TestImportTimeValidation:
    """An illegal entry or a dangling baseline must abort the catalog validation."""

    def test_dangling_baseline_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(METRIC_DEFINITIONS, 'bogus_metric', MetricEntry(baseline='quality.does.not.exist'))

        with pytest.raises(ValueError, match='unknown baseline'):
            catalog_module._validate_catalog()

    def test_illegal_entry_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # role=primary with direction=none violates the contract and must not resolve.
        monkeypatch.setitem(
            METRIC_DEFINITIONS,
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


class TestGsm8kAccuracy:
    """GSM8K's canonical accuracy definition has the expected quality contract."""

    def test_accuracy_resolves_to_primary_accuracy(self) -> None:
        semantics = METRIC_DEFINITIONS['accuracy'].resolve('accuracy')

        assert semantics.semantic_id == 'quality.accuracy.ratio'
        assert semantics.role is MetricRole.PRIMARY
        assert semantics.direction is MetricDirection.HIGHER_IS_BETTER


def test_job_bench_normalized_score_is_a_bounded_ratio() -> None:
    semantics = METRIC_DEFINITIONS['normalized_score'].resolve('normalized_score')

    assert semantics.semantic_id == 'quality.score.ratio'
    assert semantics.value_range is not None
    assert semantics.value_range.min == 0
    assert semantics.value_range.max == 1


def test_v2_registry_contains_only_canonical_non_dynamic_names() -> None:
    forbidden_names = {'score', 'overall', 'total_score'}
    for name in METRIC_DEFINITIONS:
        assert name == name.lower()
        assert not name.startswith('mean_')
        assert not name.endswith(('_s', '_ms'))
        assert all(character not in name for character in ('@', '/', ' '))
        assert name not in forbidden_names
